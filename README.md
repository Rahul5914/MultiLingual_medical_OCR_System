# 🏥 Multilingual Medical OCR System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/PyTorch-2.1+-orange?logo=pytorch" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

> **End-to-end AI pipeline** that extracts text from medical documents (prescriptions, lab reports, handwritten notes), detects the source language, translates to English, performs BERT-based Named Entity Recognition, and generates a professional PDF report — all in one click.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 🧠 **Hybrid OCR** | TrOCR (transformer) + EasyOCR + Tesseract in parallel |
| 🌐 **Multilingual** | Hindi, Bengali, Odia, Tamil, Telugu, Marathi + more |
| 🔄 **Auto-translation** | Language detection → English via Google Translate |
| 🤖 **BERT NER** | `dslim/bert-base-NER` for person, organisation, location entities |
| 💊 **Medical Extraction** | Rule-based extraction of medicines, dosages, diagnosis, patient info |
| 📄 **PDF Report** | Styled A4 report with tables, confidence scores, entity highlights |
| 🎯 **Confidence Scores** | Per-engine OCR confidence displayed in UI and PDF |
| 📂 **Batch Processing** | Upload and process multiple images simultaneously |
| 🖥️ **Streamlit UI** | Clean, responsive web interface with dark-themed header |
| 📓 **Colab Notebook** | Ready-to-run notebook with all 12 ordered cells |

---

## 🗂️ Project Structure

```
multilingual_medical_ocr/
│
├── app.py              # Streamlit web application
├── ocr_pipeline.py     # Core pipeline module (all logic)
├── notebook.ipynb      # Google Colab notebook (12 cells)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/multilingual-medical-ocr.git
cd multilingual-medical-ocr
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR (System Package)

**Ubuntu / Debian / Google Colab:**
```bash
sudo apt-get install -y tesseract-ocr \
    tesseract-ocr-hin \
    tesseract-ocr-ben \
    tesseract-ocr-ori \
    tesseract-ocr-tam \
    tesseract-ocr-tel
```

**macOS (Homebrew):**
```bash
brew install tesseract
```

**Windows:**  
Download installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH.

---

## 🖥️ Running the Streamlit App

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

The app will:
1. Load all models on first run (TrOCR, EasyOCR, BERT NER) — ~2 min
2. Present a clean upload interface
3. Process images and display results in real-time
4. Allow PDF download

---

## 📓 Running the Colab Notebook

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `notebook.ipynb`
3. Go to **Runtime → Change runtime type → T4 GPU**
4. Click **Runtime → Run all** (`Ctrl+F9`)
5. When prompted, upload a medical image
6. The PDF will auto-download when processing completes

---

## 🧠 Model Details

| Component | Model | Source |
|-----------|-------|--------|
| TrOCR | `microsoft/trocr-base-printed` | HuggingFace |
| EasyOCR | JaidedAI EasyOCR | PyPI |
| Tesseract | Tesseract 5.x | System |
| BERT NER | `dslim/bert-base-NER` | HuggingFace |
| Translation | GoogleTranslator | deep-translator |

---

## 📊 Pipeline Architecture

```
Image Input
    │
    ▼
Image Preprocessor (CLAHE + Denoise + Deskew + Sharpen)
    │
    ├──► TrOCR ──────────┐
    ├──► EasyOCR ─────────┤ Confidence-weighted combiner
    └──► Tesseract ───────┘
                          │
                          ▼
                    Combined Text
                          │
              ┌───────────┴────────────┐
              ▼                        ▼
    Language Detection         Rule-based Medical
    + Google Translation       Entity Extraction
              │                        │
              ▼                        ▼
         BERT NER                Structured Info
         (PER/ORG/LOC)        (Patient/Meds/Dx/Date)
              │
              └──────────────────────────┐
                                         ▼
                                   PDF Report (A4)
```

---

## 📋 Output PDF Sections

1. **Header** — System title, generation timestamp, detected language
2. **Patient & Prescription Details** — Extracted patient info table
3. **OCR Confidence Scores** — Per-engine scores with status indicators
4. **Extracted OCR Text** — Raw combined output
5. **English Translation** — Translated text (if source ≠ English)
6. **Medicines & Dosages** — Identified drugs and instructions
7. **Diagnosis / Findings** — Clinical keywords
8. **BERT NER Entities** — Named entities with confidence scores
9. **Footer** — Disclaimer

---

## 📸 Screenshots

> *Add your screenshots here after running the application*

| Upload & Process | Results Dashboard |
|-----------------|-------------------|
| `[Upload screenshot]` | `[Results screenshot]` |

| PDF Report | Batch Processing |
|------------|-----------------|
| `[PDF screenshot]` | `[Batch screenshot]` |

---

## ⚙️ Configuration

You can modify these constants in `ocr_pipeline.py`:

```python
# Add/remove EasyOCR languages
easyocr.Reader(["en", "hi", "bn", "ta"])  # line 91

# Adjust BERT model
"dslim/bert-base-NER"   # swap for "allenai/scibert-base" or similar

# Image scale threshold
if max(h, w) < 1000:    # line 57 — change minimum resolution
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `tesseract not found` | Install Tesseract and add to PATH |
| `CUDA out of memory` | Reduce batch size or use CPU (`device=-1`) |
| `Translation quota exceeded` | Wait 1 min (Google free tier limit) |
| `EasyOCR slow on first run` | Models are downloaded on first call |
| `Colab disconnects` | Enable GPU and check RAM usage |

---

## 📄 License

MIT License © 2024

---

## 🙏 Acknowledgements

- [Microsoft TrOCR](https://huggingface.co/microsoft/trocr-base-printed)
- [JaidedAI EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER)
- [deep-translator](https://github.com/nidhaloff/deep-translator)
- [Streamlit](https://streamlit.io/)
- [ReportLab](https://www.reportlab.com/)

---

<p align="center">Built for academic research and healthcare AI innovation 🏥</p>
