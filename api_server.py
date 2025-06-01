from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import tempfile
from main import extract_text_from_docx, extract_text_from_pdf, extract_text_from_png, extract_info_with_groq

app = FastAPI()

@app.post("/extract/")
async def extract(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        if suffix == ".docx":
            text = extract_text_from_docx(temp_path)
        elif suffix == ".pdf":
            text = extract_text_from_pdf(temp_path)
        elif suffix == ".png":
            text = extract_text_from_png(temp_path)
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file type"})

        if not text:
            return JSONResponse(status_code=400, content={"error": "Text extraction failed"})

        result = extract_info_with_groq(text)
        if result:
            return result.model_dump()
        else:
            return JSONResponse(status_code=500, content={"error": "LLM extraction failed"})

    finally:
        os.remove(temp_path)
