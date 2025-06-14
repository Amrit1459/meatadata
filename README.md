# 🧠 Document Metadata Extractor

This is a project built to extract **important metadata** from documents like contracts, agreements, or any official papers — whether they're `.docx`, `.pdf`, or even images (`.png`, `.jpg`). It combines OCR, NLP, and machine learning to pull out meaningful information like party names, dates, and agreement values.

---

## 📌 What It Does

You upload a document. It does the heavy lifting.

- 📃 Extracts text from `.docx`, `.pdf`, `.png`, `.jpg`, `.jpeg`
- 🤖 Uses OCR (EasyOCR) for images
- 🧠 Applies NLP (spaCy) for named entity recognition
- 🔍 Pulls out:
  - Agreement Value 💰
  - Agreement Start & End Dates 🗓️
  - Renewal Notice (in days)
  - Party Names 🧑‍🤝‍🧑

You can use it via:
- A **simple Flask web interface**, or
- A **FastAPI backend** for API access

---

## 🚀 How to Use

1. Clone the Repo

```bash
git clone https://github.com/Amrit1459/meatadata.git
cd document-metadata-extractor



2.INSTALL REQUIREMENT

pip install -r requirements.txt
python -m spacy download en_core_web_sm


3.DATA PREPARATION

data/
├── train.csv          # Contains file names and expected metadata
├── test.csv           
├── train/           
└── test/              


4.Run the Web App (Flask)

python main.py

