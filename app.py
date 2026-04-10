"""
app.py
=======
Streamlit front-end for the Multilingual Medical OCR System.
Run with:   streamlit run app.py
"""

import io
import os
import time
import json
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np

# ─── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Multilingual Medical OCR",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #1565c0 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(26,35,126,0.3);
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; font-weight: 700; }
    .main-header p  { margin: 0.4rem 0 0; opacity: 0.85; font-size: 1rem; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #0d47a1;
        height: 100%;
    }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #0d47a1; }
    .metric-card .label { font-size: 0.8rem; color: #607d8b; margin-top: 4px; }

    /* Confidence bar */
    .conf-bar-container { margin: 0.4rem 0; }
    .conf-label { font-size: 0.85rem; font-weight: 600; color: #37474f; }
    .conf-bar {
        height: 10px; border-radius: 5px;
        background: linear-gradient(90deg, #43a047, #66bb6a);
        display: inline-block; margin-left: 8px;
        vertical-align: middle;
    }
    .conf-val { font-size: 0.8rem; color: #546e7a; margin-left: 6px; }

    /* Entity badges */
    .entity-badge {
        display: inline-block;
        background: #e8eaf6;
        color: #3949ab;
        border-radius: 14px;
        padding: 3px 12px;
        margin: 3px;
        font-size: 0.82rem;
        font-weight: 500;
        border: 1px solid #9fa8da;
    }
    .entity-badge.med  { background:#e8f5e9; color:#2e7d32; border-color:#a5d6a7; }
    .entity-badge.diag { background:#fff3e0; color:#e65100; border-color:#ffcc80; }
    .entity-badge.ner  { background:#f3e5f5; color:#6a1b9a; border-color:#ce93d8; }

    /* Section headers */
    .section-header {
        font-size: 1rem; font-weight: 700; color: #1a237e;
        border-bottom: 2px solid #e8eaf6; padding-bottom: 4px;
        margin-top: 1rem; margin-bottom: 0.6rem;
    }

    /* Text box */
    .text-box {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.88rem;
        line-height: 1.6;
        white-space: pre-wrap;
        word-break: break-word;
        max-height: 300px;
        overflow-y: auto;
    }

    /* Download button */
    div[data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-size: 0.95rem;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
        transition: all 0.2s;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background: linear-gradient(135deg, #283593, #1565c0);
        box-shadow: 0 4px 12px rgba(13,71,161,0.4);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #f8f9ff; }

    /* Status pills */
    .status-ok   { color:#2e7d32; font-weight:600; }
    .status-warn { color:#e65100; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🏥 Multilingual Medical OCR System</h1>
        <p>Extract · Translate · Analyse · Report  |  Powered by TrOCR · EasyOCR · Tesseract · BERT NER</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Lazy import pipeline (avoids Streamlit reload issues) ────────────────────
@st.cache_resource(show_spinner="🔄 Loading AI models (first run may take a few minutes) …")
def load_pipeline():
    from ocr_pipeline import (
        ModelLoader, ImagePreprocessor, OCREngine,
        combine_ocr_results, detect_and_translate,
        extract_medical_info_rules, extract_entities_bert,
        generate_pdf_report,
    )
    # Pre-warm all models
    ModelLoader.load_trocr()
    ModelLoader.load_easyocr()
    ModelLoader.load_ner()
    return {
        "ImagePreprocessor": ImagePreprocessor,
        "OCREngine": OCREngine,
        "combine_ocr_results": combine_ocr_results,
        "detect_and_translate": detect_and_translate,
        "extract_medical_info_rules": extract_medical_info_rules,
        "extract_entities_bert": extract_entities_bert,
        "generate_pdf_report": generate_pdf_report,
    }


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    use_trocr = st.checkbox("TrOCR (Microsoft)", value=True)
    use_easyocr = st.checkbox("EasyOCR", value=True)
    use_tesseract = st.checkbox("Tesseract", value=True)

    st.markdown("---")
    st.markdown("## 🌐 Target Language")
    target_lang = st.selectbox(
        "Translate output to:",
        ["English", "Hindi", "Bengali", "Tamil", "Telugu"],
        index=0,
    )

    st.markdown("---")
    st.markdown("## ℹ️ About")
    st.markdown(
        """
        **Multilingual Medical OCR**
        - 🧠 TrOCR — transformer OCR
        - 📖 EasyOCR — multi-language OCR
        - 🔍 Tesseract — classical OCR
        - 🤖 BERT NER — entity recognition
        - 🌐 Google Translate — auto-translation
        - 📄 ReportLab — PDF generation
        """
    )
    st.markdown("---")
    st.caption("v1.0 · For research & educational use")

# ─── Main content ─────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_help = st.tabs(
    ["📷 Single Image", "📂 Batch Processing", "📘 How to Use"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE IMAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab_single:
    col_upload, col_preview = st.columns([1, 1], gap="medium")

    with col_upload:
        st.markdown('<div class="section-header">📤 Upload Medical Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Supported: JPG, PNG, BMP, TIFF",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            key="single_upload",
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            with col_preview:
                st.markdown('<div class="section-header">🖼️ Uploaded Image</div>', unsafe_allow_html=True)
                st.image(image, use_container_width=True, caption=uploaded_file.name)

            st.markdown("---")
            run_btn = st.button("🚀 Run OCR Pipeline", type="primary", use_container_width=True)

            if run_btn:
                pipeline = load_pipeline()

                with st.status("⚙️ Processing …", expanded=True) as status:
                    t0 = time.time()

                    st.write("🖼️ Pre-processing image …")
                    preprocessor = pipeline["ImagePreprocessor"]
                    processed_gray = preprocessor.preprocess(image)
                    pil_processed = preprocessor.to_pil(processed_gray)

                    # ── OCR ─────────────────────────────────────────────
                    ocr_engine = pipeline["OCREngine"]
                    trocr_res = easyocr_res = tesseract_res = None

                    if use_trocr:
                        st.write("🧠 Running TrOCR …")
                        trocr_res = ocr_engine.run_trocr(pil_processed)

                    if use_easyocr:
                        st.write("📖 Running EasyOCR …")
                        easyocr_res = ocr_engine.run_easyocr(pil_processed)

                    if use_tesseract:
                        st.write("🔍 Running Tesseract …")
                        tesseract_res = ocr_engine.run_tesseract(processed_gray)

                    # Fallback empty results
                    empty = {"text": "", "confidence": 0.0, "engine": "—"}
                    trocr_res = trocr_res or empty
                    easyocr_res = easyocr_res or empty
                    tesseract_res = tesseract_res or empty

                    st.write("🔀 Combining OCR outputs …")
                    combined_text = pipeline["combine_ocr_results"](trocr_res, easyocr_res, tesseract_res)

                    st.write("🌐 Detecting language & translating …")
                    translation = pipeline["detect_and_translate"](combined_text)

                    work_text = translation["translated"] or combined_text

                    st.write("💊 Extracting medical entities …")
                    medical_info = pipeline["extract_medical_info_rules"](work_text)

                    st.write("🤖 Running BERT NER …")
                    bert_ents = pipeline["extract_entities_bert"](work_text)

                    st.write("📄 Generating PDF report …")
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        pdf_path = tmp.name
                    pipeline["generate_pdf_report"](
                        image_input=uploaded_file.name,
                        ocr_text=combined_text,
                        translated_text=translation["translated"],
                        detected_language=translation["language"],
                        medical_info=medical_info,
                        bert_entities=bert_ents,
                        trocr_conf=trocr_res["confidence"],
                        easyocr_conf=easyocr_res["confidence"],
                        tesseract_conf=tesseract_res["confidence"],
                        output_path=pdf_path,
                    )
                    elapsed = time.time() - t0
                    status.update(label=f"✅ Done in {elapsed:.1f}s", state="complete")

                # ── RESULTS ─────────────────────────────────────────────
                st.markdown("---")
                st.markdown("## 📊 Results")

                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(
                        f'<div class="metric-card"><div class="value">{len(combined_text)}</div>'
                        f'<div class="label">Characters Extracted</div></div>',
                        unsafe_allow_html=True,
                    )
                with m2:
                    st.markdown(
                        f'<div class="metric-card"><div class="value">{len(medical_info.get("Medicines",[]))}</div>'
                        f'<div class="label">Medicines Found</div></div>',
                        unsafe_allow_html=True,
                    )
                with m3:
                    st.markdown(
                        f'<div class="metric-card"><div class="value">{sum(len(v) for v in bert_ents.values())}</div>'
                        f'<div class="label">BERT Entities</div></div>',
                        unsafe_allow_html=True,
                    )
                with m4:
                    st.markdown(
                        f'<div class="metric-card"><div class="value">{translation["language"]}</div>'
                        f'<div class="label">Detected Language</div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # ── OCR Confidence ───────────────────────────────────────
                st.markdown('<div class="section-header">🎯 OCR Engine Confidence</div>', unsafe_allow_html=True)
                conf_cols = st.columns(3)
                for col, res, icon in zip(
                    conf_cols,
                    [trocr_res, easyocr_res, tesseract_res],
                    ["🧠 TrOCR", "📖 EasyOCR", "🔍 Tesseract"],
                ):
                    with col:
                        conf = res["confidence"]
                        color = "#43a047" if conf > 0.6 else "#fb8c00" if conf > 0.3 else "#e53935"
                        st.markdown(
                            f"<b>{icon}</b><br>"
                            f'<div style="background:#f5f5f5;border-radius:8px;padding:8px;margin-top:4px;">'
                            f'<div style="height:10px;border-radius:5px;background:{color};'
                            f'width:{min(conf*100,100):.0f}%;"></div>'
                            f'<span style="font-size:0.85rem;color:{color};font-weight:700;">'
                            f'{conf*100:.1f}%</span></div>',
                            unsafe_allow_html=True,
                        )

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Text results ──────────────────────────────────────────
                r1, r2 = st.columns(2)
                with r1:
                    st.markdown('<div class="section-header">📄 Extracted Text</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="text-box">{combined_text or "No text extracted."}</div>',
                        unsafe_allow_html=True,
                    )
                with r2:
                    st.markdown(
                        f'<div class="section-header">🌐 English Translation '
                        f'<span style="font-size:0.8rem;font-weight:400;">'
                        f'(from {translation["language"]})</span></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="text-box">{translation["translated"] or "—"}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Medical Entities ──────────────────────────────────────
                e1, e2 = st.columns(2)
                with e1:
                    st.markdown('<div class="section-header">💊 Identified Medicines</div>', unsafe_allow_html=True)
                    meds = medical_info.get("Medicines", [])
                    if meds:
                        html = "".join(
                            f'<span class="entity-badge med">💊 {m}</span>' for m in meds
                        )
                        st.markdown(html, unsafe_allow_html=True)
                    else:
                        st.info("No specific medicines identified.")

                    st.markdown('<div class="section-header" style="margin-top:1rem;">🔬 Diagnosis / Findings</div>', unsafe_allow_html=True)
                    diags = medical_info.get("Diagnosis", [])
                    if diags:
                        html = "".join(
                            f'<span class="entity-badge diag">🔬 {d}</span>' for d in diags
                        )
                        st.markdown(html, unsafe_allow_html=True)
                    else:
                        st.info("No diagnosis keywords found.")

                with e2:
                    st.markdown('<div class="section-header">🤖 BERT NER Entities</div>', unsafe_allow_html=True)
                    if bert_ents:
                        for label, items in bert_ents.items():
                            st.markdown(f"**{label}**")
                            parts = []
                            for i in items:
                                sc = i["score"]
                                wd = i["word"]
                                parts.append(
                                    f'<span class="entity-badge ner" title="conf: {sc}">'
                                    f'{wd} <small>({sc:.2f})</small></span>'
                                )
                            st.markdown("".join(parts), unsafe_allow_html=True)
                    else:
                        st.info("No entities extracted by BERT.")

                # ── Patient details ───────────────────────────────────────
                st.markdown("---")
                st.markdown('<div class="section-header">🧑‍⚕️ Patient & Prescription Details</div>', unsafe_allow_html=True)
                det1, det2, det3 = st.columns(3)
                with det1:
                    st.metric("Patient Name", medical_info.get("Patient") or "—")
                    st.metric("Date of Birth", medical_info.get("DOB") or "—")
                with det2:
                    st.metric("Prescription Date", medical_info.get("Date") or "—")
                    st.metric("Doctor", medical_info.get("Doctor") or "—")
                with det3:
                    st.metric("Hospital / Clinic", medical_info.get("Hospital") or "—")
                    st.metric("Elapsed Time", f"{elapsed:.1f}s")

                # ── PDF Download ─────────────────────────────────────────
                st.markdown("---")
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="📥 Download Full PDF Report",
                    data=pdf_bytes,
                    file_name="medical_ocr_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

                # ── Raw JSON ─────────────────────────────────────────────
                with st.expander("🔍 View Raw Extracted Data (JSON)"):
                    export = {
                        "ocr_text": combined_text,
                        "translation": translation,
                        "medical_info": {k: v for k, v in medical_info.items()
                                         if k != "Raw_Lines"},
                        "bert_entities": bert_ents,
                        "confidence": {
                            "trocr": round(trocr_res["confidence"], 3),
                            "easyocr": round(easyocr_res["confidence"], 3),
                            "tesseract": round(tesseract_res["confidence"], 3),
                        },
                    }
                    st.json(export)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### 📂 Batch Image Processing")
    st.info("Upload multiple images and process them all at once. Each image will get its own PDF report.")

    batch_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        accept_multiple_files=True,
        key="batch_upload",
    )

    if batch_files:
        st.write(f"**{len(batch_files)} file(s) selected.**")
        # Preview grid
        cols = st.columns(min(len(batch_files), 4))
        for i, f in enumerate(batch_files):
            with cols[i % 4]:
                st.image(Image.open(f), caption=f.name, use_container_width=True)

        if st.button("🚀 Process All Images", type="primary", use_container_width=True):
            pipeline = load_pipeline()
            results_summary = []
            all_pdfs = {}

            progress = st.progress(0, text="Processing …")
            for idx, f in enumerate(batch_files):
                img = Image.open(f)
                preprocessor = pipeline["ImagePreprocessor"]
                processed_gray = preprocessor.preprocess(img)
                pil_processed = preprocessor.to_pil(processed_gray)

                trocr_res = pipeline["OCREngine"].run_trocr(pil_processed)
                easyocr_res = pipeline["OCREngine"].run_easyocr(pil_processed)
                tesseract_res = pipeline["OCREngine"].run_tesseract(processed_gray)

                combined = pipeline["combine_ocr_results"](trocr_res, easyocr_res, tesseract_res)
                translation = pipeline["detect_and_translate"](combined)
                work_text = translation["translated"] or combined
                medical_info = pipeline["extract_medical_info_rules"](work_text)
                bert_ents = pipeline["extract_entities_bert"](work_text)

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    pdf_path = tmp.name
                pipeline["generate_pdf_report"](
                    image_input=f.name,
                    ocr_text=combined,
                    translated_text=translation["translated"],
                    detected_language=translation["language"],
                    medical_info=medical_info,
                    bert_entities=bert_ents,
                    trocr_conf=trocr_res["confidence"],
                    easyocr_conf=easyocr_res["confidence"],
                    tesseract_conf=tesseract_res["confidence"],
                    output_path=pdf_path,
                )
                all_pdfs[f.name] = pdf_path
                results_summary.append({
                    "File": f.name,
                    "Language": translation["language"],
                    "Chars": len(combined),
                    "Medicines": len(medical_info.get("Medicines", [])),
                })
                progress.progress((idx + 1) / len(batch_files),
                                  text=f"Processed {idx+1}/{len(batch_files)}: {f.name}")

            st.success("✅ All images processed!")
            st.dataframe(results_summary, use_container_width=True)

            st.markdown("### 📥 Download Reports")
            for fname, pdf_path in all_pdfs.items():
                with open(pdf_path, "rb") as pf:
                    st.download_button(
                        label=f"📥 {fname} → PDF",
                        data=pf.read(),
                        file_name=f"{Path(fname).stem}_report.pdf",
                        mime="application/pdf",
                    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HELP
# ══════════════════════════════════════════════════════════════════════════════
with tab_help:
    st.markdown(
        """
        ## 📘 How to Use

        ### Single Image Mode
        1. Select **📷 Single Image** tab.
        2. Upload a medical prescription, lab report, or handwritten document.
        3. Choose OCR engines in the sidebar (all three recommended).
        4. Click **🚀 Run OCR Pipeline**.
        5. Review extracted text, translation, entities, and download the PDF.

        ### Batch Mode
        1. Select **📂 Batch Processing** tab.
        2. Upload multiple images at once.
        3. Click **🚀 Process All Images**.
        4. Download individual PDF reports for each image.

        ---

        ### Supported Languages (OCR)
        | Language | Code | Notes |
        |----------|------|-------|
        | English  | en   | Primary |
        | Hindi    | hi   | EasyOCR + Tesseract |
        | Bengali  | bn   | EasyOCR + Tesseract |
        | Odia     | or   | Tesseract |
        | Tamil    | ta   | EasyOCR + Tesseract |

        ---

        ### Models Used
        | Component | Model | Purpose |
        |-----------|-------|---------|
        | TrOCR | `microsoft/trocr-base-printed` | Transformer OCR |
        | EasyOCR | `jaided-ai/easyocr` | Multi-language OCR |
        | Tesseract | `tesseract-ocr` | Classical OCR fallback |
        | NER | `dslim/bert-base-NER` | Named Entity Recognition |
        | Translation | `deep-translator` + Google | Auto-translation |

        ---

        ### Tips for Best Results
        - Use clear, high-resolution images (≥ 300 DPI).
        - Ensure good lighting and minimal shadows.
        - Flatten curved or wrinkled documents before scanning.
        - For handwritten text, TrOCR performs best.
        """
    )
