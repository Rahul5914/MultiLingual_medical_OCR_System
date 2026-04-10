"""
ocr_pipeline.py
================
Multilingual Medical OCR Pipeline
Supports: TrOCR + EasyOCR + Tesseract | BERT NER | Translation | PDF Report
"""

import os
import re
import io
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# OCR
import easyocr
import pytesseract

# HuggingFace
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    pipeline as hf_pipeline,
)
import torch

# Translation / Language Detection
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 1. MODEL LOADER
# ─────────────────────────────────────────────

class ModelLoader:
    """Singleton-style loader so models are loaded only once."""

    _trocr_processor = None
    _trocr_model = None
    _easyocr_reader = None
    _ner_pipeline = None
    _device = None

    @classmethod
    def get_device(cls):
        if cls._device is None:
            cls._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {cls._device}")
        return cls._device

    @classmethod
    def load_trocr(cls):
        if cls._trocr_processor is None:
            logger.info("Loading TrOCR model …")
            cls._trocr_processor = TrOCRProcessor.from_pretrained(
                "microsoft/trocr-base-printed"
            )
            cls._trocr_model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-base-printed"
            ).to(cls.get_device())
            cls._trocr_model.eval()
            logger.info("TrOCR loaded ✓")
        return cls._trocr_processor, cls._trocr_model

    @classmethod
    def load_easyocr(cls):
        if cls._easyocr_reader is None:
            logger.info("Loading EasyOCR …")
            cls._easyocr_reader = easyocr.Reader(
                ["en", "hi"],   # add more language codes as needed
                gpu=torch.cuda.is_available(),
            )
            logger.info("EasyOCR loaded ✓")
        return cls._easyocr_reader

    @classmethod
    def load_ner(cls):
        if cls._ner_pipeline is None:
            logger.info("Loading BERT NER …")
            cls._ner_pipeline = hf_pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple",
                device=0 if cls.get_device() == "cuda" else -1,
            )
            logger.info("BERT NER loaded ✓")
        return cls._ner_pipeline


# ─────────────────────────────────────────────
# 2. IMAGE PREPROCESSOR
# ─────────────────────────────────────────────

class ImagePreprocessor:
    """Advanced preprocessing: CLAHE + denoising + adaptive thresholding."""

    @staticmethod
    def preprocess(image_input) -> np.ndarray:
        """
        Accept PIL Image, file path, or numpy array.
        Returns a clean grayscale numpy array ready for OCR.
        """
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
        elif isinstance(image_input, Image.Image):
            img = cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        # ── Upscale if small ──────────────────────────────────────────────
        h, w = img.shape[:2]
        if max(h, w) < 1000:
            scale = 1500 / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)

        # ── Grayscale ─────────────────────────────────────────────────────
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── CLAHE (contrast-limited adaptive histogram equalisation) ──────
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # ── Gaussian denoising ────────────────────────────────────────────
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # ── Sharpening ────────────────────────────────────────────────────
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)

        # ── Deskew ────────────────────────────────────────────────────────
        gray = ImagePreprocessor._deskew(gray)

        return gray

    @staticmethod
    def _deskew(gray: np.ndarray) -> np.ndarray:
        """Correct skew using Hough lines."""
        try:
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
            if lines is None:
                return gray
            angles = []
            for line in lines[:20]:
                rho, theta = line[0]
                if theta < np.pi / 4 or theta > 3 * np.pi / 4:
                    angles.append(theta - np.pi / 2)
            if not angles:
                return gray
            median_angle = np.median(angles) * 180 / np.pi
            if abs(median_angle) > 0.5:
                (h, w) = gray.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                gray = cv2.warpAffine(
                    gray, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
        except Exception:
            pass
        return gray

    @staticmethod
    def to_pil(processed: np.ndarray) -> Image.Image:
        return Image.fromarray(processed)


# ─────────────────────────────────────────────
# 3. OCR ENGINES
# ─────────────────────────────────────────────

class OCREngine:
    """Wraps TrOCR, EasyOCR and Tesseract with confidence scoring."""

    # ---------- TrOCR ----------

    @staticmethod
    def run_trocr(pil_image: Image.Image) -> dict:
        processor, model = ModelLoader.load_trocr()
        device = ModelLoader.get_device()
        try:
            rgb = pil_image.convert("RGB")
            pixel_values = processor(images=rgb, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_new_tokens=512)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            confidence = 0.85 if text else 0.0   # TrOCR doesn't expose per-token probs easily
            return {"text": text, "confidence": confidence, "engine": "TrOCR"}
        except Exception as e:
            logger.warning(f"TrOCR failed: {e}")
            return {"text": "", "confidence": 0.0, "engine": "TrOCR"}

    # ---------- EasyOCR ----------

    @staticmethod
    def run_easyocr(pil_image: Image.Image) -> dict:
        reader = ModelLoader.load_easyocr()
        try:
            img_array = np.array(pil_image.convert("RGB"))
            results = reader.readtext(img_array, detail=1)
            if not results:
                return {"text": "", "confidence": 0.0, "engine": "EasyOCR", "details": []}
            lines, confs = [], []
            for (_, text, conf) in results:
                lines.append(text)
                confs.append(conf)
            combined_text = "\n".join(lines)
            avg_conf = float(np.mean(confs)) if confs else 0.0
            return {
                "text": combined_text,
                "confidence": avg_conf,
                "engine": "EasyOCR",
                "details": [{"text": t, "confidence": c} for (_, t, c) in results],
            }
        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")
            return {"text": "", "confidence": 0.0, "engine": "EasyOCR", "details": []}

    # ---------- Tesseract ----------

    @staticmethod
    def run_tesseract(processed_gray: np.ndarray) -> dict:
        try:
            custom_config = r"--oem 3 --psm 6 -l eng+hin+ben+ori"
            data = pytesseract.image_to_data(
                processed_gray,
                config=custom_config,
                output_type=pytesseract.Output.DICT,
            )
            words, confs = [], []
            for i, word in enumerate(data["text"]):
                conf = int(data["conf"][i])
                if word.strip() and conf > 10:
                    words.append(word)
                    confs.append(conf)
            text = " ".join(words)
            avg_conf = float(np.mean(confs)) / 100.0 if confs else 0.0
            return {"text": text, "confidence": avg_conf, "engine": "Tesseract"}
        except Exception as e:
            logger.warning(f"Tesseract failed: {e}")
            return {"text": "", "confidence": 0.0, "engine": "Tesseract"}


# ─────────────────────────────────────────────
# 4. TEXT COMBINER
# ─────────────────────────────────────────────

def combine_ocr_results(trocr: dict, easyocr_res: dict, tesseract: dict) -> str:
    """
    Weighted combination: pick the highest-confidence result as primary,
    then supplement with additional tokens from the others.
    """
    results = [trocr, easyocr_res, tesseract]
    results.sort(key=lambda x: x["confidence"], reverse=True)

    # Primary text (highest confidence)
    primary = results[0]["text"].strip()

    # Collect extra words from the other engines not already in primary
    primary_tokens = set(primary.lower().split())
    extras = []
    for r in results[1:]:
        for token in r["text"].split():
            if token.lower() not in primary_tokens and len(token) > 2:
                extras.append(token)
                primary_tokens.add(token.lower())

    combined = primary
    if extras:
        combined += "\n" + " ".join(extras)

    return combined.strip()


# ─────────────────────────────────────────────
# 5. LANGUAGE DETECTION & TRANSLATION
# ─────────────────────────────────────────────

LANG_MAP = {
    "hi": "Hindi",
    "bn": "Bengali",
    "or": "Odia",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "en": "English",
}


def detect_and_translate(text: str) -> dict:
    """Detect language and translate to English if needed."""
    if not text.strip():
        return {"original": text, "translated": text, "language": "unknown", "lang_code": "en"}

    try:
        lang_code = detect(text)
    except LangDetectException:
        lang_code = "en"

    lang_name = LANG_MAP.get(lang_code, lang_code.upper())

    if lang_code == "en":
        return {
            "original": text,
            "translated": text,
            "language": "English",
            "lang_code": "en",
        }

    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        translated = text

    return {
        "original": text,
        "translated": translated,
        "language": lang_name,
        "lang_code": lang_code,
    }


# ─────────────────────────────────────────────
# 6. MEDICAL ENTITY EXTRACTION (Rule-based)
# ─────────────────────────────────────────────

# Common medical keywords for rule-based extraction
MEDICINE_PATTERNS = [
    r"\b(?:Tab|Cap|Syp|Inj|Oint|Drop|Syr)\b\.?\s*\w+",
    r"\b\w+(?:cin|mycin|cillin|zole|pril|sartan|olol|statin|pam|oxacin|mab)\b",
    r"\b\w+\s+\d+\s*(?:mg|mcg|ml|g|IU)\b",
    r"\b(?:Amoxicillin|Paracetamol|Metformin|Aspirin|Ibuprofen|Azithromycin|"
    r"Ciprofloxacin|Omeprazole|Atorvastatin|Metoprolol|Amlodipine|"
    r"Cetirizine|Pantoprazole|Dolo|Calpol|Augmentin|Crocin)\b",
]

DOSAGE_PATTERNS = [
    r"\b\d+\s*(?:mg|mcg|ml|g|IU)\b",
    r"\b(?:once|twice|thrice|OD|BD|TDS|QID|PRN|SOS|HS|AC|PC)\b",
    r"\b(?:1|2|3|4)-(?:0|1)-(?:0|1)\b",
    r"\b\d+\s*tab(?:let)?s?\b",
    r"\b\d+\s*times?\s*(?:a\s*)?day\b",
    r"\bfor\s+\d+\s*days?\b",
]

DIAGNOSIS_PATTERNS = [
    r"(?:Diagnosis|Dx|Impression|Findings?|Assessment)\s*:?\s*(.+)",
    r"\b(?:fever|hypertension|diabetes|infection|pneumonia|bronchitis|"
    r"asthma|anaemia|anemia|UTI|URTI|cold|cough|flu|malaria|typhoid|"
    r"dengue|gastritis|migraine|arthritis|fracture|allergy)\b",
]


def extract_medical_info_rules(text: str) -> dict:
    """Rule-based extraction of medical entities from text."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    info = {
        "Patient": "",
        "DOB": "",
        "Date": "",
        "Doctor": "",
        "Hospital": "",
        "Medicines": [],
        "Dosages": [],
        "Diagnosis": [],
        "Raw_Lines": lines,
    }

    full_text = text

    # Patient name
    patient_match = re.search(
        r"(?:Patient|Name|Pt\.?)\s*[:\-]?\s*([A-Z][a-zA-Z\s\.]{2,40})", full_text
    )
    if patient_match:
        info["Patient"] = patient_match.group(1).strip()

    # DOB
    dob_match = re.search(
        r"(?:DOB|Date\s*of\s*Birth)\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        full_text,
    )
    if dob_match:
        info["DOB"] = dob_match.group(1).strip()

    # Date
    date_match = re.search(
        r"(?:Date|Dt\.?)\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})", full_text
    )
    if date_match:
        info["Date"] = date_match.group(1).strip()

    # Doctor
    doctor_match = re.search(
        r"(?:Dr\.?|Doctor|Physician)\s*[:\-]?\s*([A-Z][a-zA-Z\s\.]{2,40})", full_text
    )
    if doctor_match:
        info["Doctor"] = doctor_match.group(1).strip()

    # Medicines
    medicines_found = set()
    for pattern in MEDICINE_PATTERNS:
        for match in re.finditer(pattern, full_text, re.IGNORECASE):
            med = match.group(0).strip()
            if len(med) > 2:
                medicines_found.add(med)
    info["Medicines"] = sorted(medicines_found)

    # Dosages
    dosages_found = set()
    for pattern in DOSAGE_PATTERNS:
        for match in re.finditer(pattern, full_text, re.IGNORECASE):
            dosages_found.add(match.group(0).strip())
    info["Dosages"] = sorted(dosages_found)

    # Diagnosis
    diagnoses_found = set()
    for pattern in DIAGNOSIS_PATTERNS:
        for match in re.finditer(pattern, full_text, re.IGNORECASE):
            dx = match.group(0).strip()
            if len(dx) > 2:
                diagnoses_found.add(dx)
    info["Diagnosis"] = sorted(diagnoses_found)

    return info


# ─────────────────────────────────────────────
# 7. BERT NER EXTRACTION
# ─────────────────────────────────────────────

def extract_entities_bert(text: str) -> dict:
    """Run BERT NER and return grouped entities."""
    if not text.strip():
        return {}

    ner = ModelLoader.load_ner()
    try:
        raw_entities = ner(text[:512])   # BERT max token limit
    except Exception as e:
        logger.warning(f"BERT NER failed: {e}")
        return {}

    grouped: dict = {}
    for ent in raw_entities:
        label = ent.get("entity_group", ent.get("entity", "MISC"))
        word = ent.get("word", "").strip()
        score = round(float(ent.get("score", 0.0)), 3)
        if not word or len(word) < 2:
            continue
        if label not in grouped:
            grouped[label] = []
        grouped[label].append({"word": word, "score": score})

    return grouped


# ─────────────────────────────────────────────
# 8. PDF REPORT GENERATOR
# ─────────────────────────────────────────────

def generate_pdf_report(
    image_path: str,
    ocr_text: str,
    translated_text: str,
    detected_language: str,
    medical_info: dict,
    bert_entities: dict,
    trocr_conf: float,
    easyocr_conf: float,
    tesseract_conf: float,
    output_path: str = "medical_report.pdf",
) -> str:
    """Generate a styled A4 PDF report."""

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    story = []

    # ── Custom styles ─────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=18,
        textColor=colors.HexColor("#1a237e"),
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#546e7a"),
        spaceAfter=12,
        alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "SectionStyle",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#0d47a1"),
        spaceBefore=12,
        spaceAfter=4,
        borderPad=4,
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#212121"),
    )
    tag_style = ParagraphStyle(
        "TagStyle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#1b5e20"),
        backColor=colors.HexColor("#e8f5e9"),
        borderPad=3,
        leading=14,
    )

    # ── Header ────────────────────────────────────────────────────────────
    story.append(Paragraph("🏥 Multilingual Medical OCR Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M:%S')} | "
        f"Source Language: <b>{detected_language}</b>",
        subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a237e")))
    story.append(Spacer(1, 0.4 * cm))

    # ── Patient Info ──────────────────────────────────────────────────────
    story.append(Paragraph("Patient & Prescription Details", section_style))
    patient_data = [
        ["Field", "Value"],
        ["Patient Name", medical_info.get("Patient", "—") or "—"],
        ["Date of Birth", medical_info.get("DOB", "—") or "—"],
        ["Prescription Date", medical_info.get("Date", "—") or "—"],
        ["Doctor / Physician", medical_info.get("Doctor", "—") or "—"],
        ["Hospital / Clinic", medical_info.get("Hospital", "—") or "—"],
    ]
    pt = Table(patient_data, colWidths=[5 * cm, 12 * cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a237e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f5f5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#e8eaf6")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdbdbd")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.4 * cm))

    # ── OCR Confidence ────────────────────────────────────────────────────
    story.append(Paragraph("OCR Engine Confidence Scores", section_style))
    conf_data = [
        ["Engine", "Confidence", "Status"],
        ["TrOCR (Microsoft)", f"{trocr_conf * 100:.1f}%",
         "✓ Good" if trocr_conf > 0.6 else "⚠ Low"],
        ["EasyOCR", f"{easyocr_conf * 100:.1f}%",
         "✓ Good" if easyocr_conf > 0.6 else "⚠ Low"],
        ["Tesseract", f"{tesseract_conf * 100:.1f}%",
         "✓ Good" if tesseract_conf > 0.6 else "⚠ Low"],
    ]
    ct = Table(conf_data, colWidths=[6 * cm, 5 * cm, 6 * cm])
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d47a1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#e3f2fd")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdbdbd")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
    ]))
    story.append(ct)
    story.append(Spacer(1, 0.4 * cm))

    # ── Extracted Text ────────────────────────────────────────────────────
    story.append(Paragraph("Extracted OCR Text (Combined)", section_style))
    safe_ocr_text = ocr_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    story.append(Paragraph(safe_ocr_text or "No text extracted.", body_style))
    story.append(Spacer(1, 0.3 * cm))

    # ── Translated Text ───────────────────────────────────────────────────
    if detected_language != "English":
        story.append(Paragraph(f"English Translation (from {detected_language})", section_style))
        safe_trans = translated_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe_trans or "—", body_style))
        story.append(Spacer(1, 0.3 * cm))

    # ── Medicines ─────────────────────────────────────────────────────────
    story.append(Paragraph("Identified Medicines & Dosages", section_style))
    medicines = medical_info.get("Medicines", [])
    dosages = medical_info.get("Dosages", [])

    if medicines or dosages:
        med_rows = [["#", "Medicine / Drug", "Dosage Information"]]
        max_rows = max(len(medicines), len(dosages), 1)
        for i in range(max_rows):
            med = medicines[i] if i < len(medicines) else "—"
            dos = dosages[i] if i < len(dosages) else "—"
            med_rows.append([str(i + 1), med, dos])
        mt = Table(med_rows, colWidths=[1 * cm, 8 * cm, 8 * cm])
        mt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e7d32")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f8e9")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#a5d6a7")),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(mt)
    else:
        story.append(Paragraph("No specific medicines identified by rule-based extraction.", body_style))
    story.append(Spacer(1, 0.4 * cm))

    # ── Diagnosis ─────────────────────────────────────────────────────────
    story.append(Paragraph("Diagnosis / Clinical Findings", section_style))
    diagnoses = medical_info.get("Diagnosis", [])
    if diagnoses:
        for dx in diagnoses:
            story.append(Paragraph(f"• {dx}", tag_style))
    else:
        story.append(Paragraph("No specific diagnosis keywords identified.", body_style))
    story.append(Spacer(1, 0.4 * cm))

    # ── BERT NER ─────────────────────────────────────────────────────────
    story.append(Paragraph("BERT Named Entity Recognition (NER)", section_style))
    if bert_entities:
        ner_rows = [["Entity Type", "Identified Entities", "Avg Confidence"]]
        for label, items in bert_entities.items():
            words = ", ".join([i["word"] for i in items])
            avg = np.mean([i["score"] for i in items])
            ner_rows.append([label, words, f"{avg:.3f}"])
        nt = Table(ner_rows, colWidths=[3.5 * cm, 11 * cm, 2.5 * cm])
        nt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a148c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3e5f5")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ce93d8")),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(nt)
    else:
        story.append(Paragraph("No named entities extracted.", body_style))

    # ── Footer ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.6 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#90a4ae")))
    story.append(Paragraph(
        "<i>Generated by Multilingual Medical OCR System | "
        "For informational purposes only. Not a substitute for professional medical advice.</i>",
        ParagraphStyle("footer", parent=styles["Normal"], fontSize=7,
                       textColor=colors.grey, alignment=TA_CENTER),
    ))

    doc.build(story)
    logger.info(f"PDF saved → {output_path}")
    return output_path


# ─────────────────────────────────────────────
# 9. MASTER PIPELINE
# ─────────────────────────────────────────────

def run_full_pipeline(image_input, output_pdf_path: str = "medical_report.pdf") -> dict:
    """
    End-to-end pipeline: image → OCR → translate → NER → PDF.

    Parameters
    ----------
    image_input : str | Path | PIL.Image | np.ndarray
    output_pdf_path : str  — path for the generated PDF

    Returns
    -------
    dict with all results
    """
    start = time.time()
    print("\n🚀 Starting Full Medical OCR Pipeline …\n")

    # ── Preprocess ─────────────────────────────────────────────────────
    processed_gray = ImagePreprocessor.preprocess(image_input)
    pil_processed = ImagePreprocessor.to_pil(processed_gray)

    # ── OCR ────────────────────────────────────────────────────────────
    trocr_result = OCREngine.run_trocr(pil_processed)
    print(f"  🧠 TrOCR  → {trocr_result['confidence']:.2f} conf")

    easyocr_result = OCREngine.run_easyocr(pil_processed)
    print(f"  📖 EasyOCR → {easyocr_result['confidence']:.2f} conf")

    tesseract_result = OCREngine.run_tesseract(processed_gray)
    print(f"  🔍 Tesseract → {tesseract_result['confidence']:.2f} conf")

    combined_text = combine_ocr_results(trocr_result, easyocr_result, tesseract_result)
    print(f"\n📄 Combined OCR text ({len(combined_text)} chars)")

    # ── Translation ────────────────────────────────────────────────────
    translation_result = detect_and_translate(combined_text)
    print(f"  🌐 Language detected: {translation_result['language']}")

    # ── Rule-based extraction ──────────────────────────────────────────
    work_text = translation_result["translated"] or combined_text
    medical_info = extract_medical_info_rules(work_text)
    print(f"  💊 Medicines found: {len(medical_info['Medicines'])}")

    # ── BERT NER ────────────────────────────────────────────────────────
    bert_entities = extract_entities_bert(work_text)
    print(f"  🤖 BERT entity groups: {list(bert_entities.keys())}")

    # ── PDF ────────────────────────────────────────────────────────────
    pdf_path = generate_pdf_report(
        image_input=str(image_input) if not isinstance(image_input, (str, Path)) else image_input,
        ocr_text=combined_text,
        translated_text=translation_result["translated"],
        detected_language=translation_result["language"],
        medical_info=medical_info,
        bert_entities=bert_entities,
        trocr_conf=trocr_result["confidence"],
        easyocr_conf=easyocr_result["confidence"],
        tesseract_conf=tesseract_result["confidence"],
        output_path=output_pdf_path,
    )

    elapsed = time.time() - start
    print(f"\n✅ Pipeline complete in {elapsed:.1f}s  |  PDF → {pdf_path}\n")

    return {
        "combined_text": combined_text,
        "trocr": trocr_result,
        "easyocr": easyocr_result,
        "tesseract": tesseract_result,
        "translation": translation_result,
        "medical_info": medical_info,
        "bert_entities": bert_entities,
        "pdf_path": pdf_path,
        "elapsed_seconds": round(elapsed, 2),
    }
