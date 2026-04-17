import io
import requests
import google.generativeai as genai
import PyPDF2
from bs4 import BeautifulSoup
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from sentence_transformers import SentenceTransformer, util

# 1. Uygulama ve Model Başlatma
app = FastAPI(
    title="Resumise AI Engine",
    description="Semantik CV Analizi ve Kariyer Koçluğu API Servisi",
    version="1.1.0"
)

# Semantic-Search Modeli
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Gemini Yapay Zeka Ayarları 
# NOT: Güvenlik için daha sonra .env'ye geçebiliriz, ama şu an çalışması için buraya ekliyoruz.
GEMINI_API_KEY = "AIzaSyAEXYL5rCHkuTXponb9eQ_RxU1k9vUF6IM" 
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-flash-latest')

# 2. Yardımcı Fonksiyonlar

def linkten_metin_cek(url: str):
    """Linkteki iş ilanı içeriğini temizleyerek çeker."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        yanit = requests.get(url, headers=headers, timeout=15)
        yanit.raise_for_status()
        
        soup = BeautifulSoup(yanit.text, 'html.parser')
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
            
        metin = soup.get_text(separator=' ')
        temiz_metin = ' '.join(metin.split())
        return temiz_metin if len(temiz_metin) > 100 else "İş ilanı içeriği kazınamadı."
    except Exception as e:
        return f"Hata: İş ilanı okunamadı. ({str(e)})"

def pdf_metin_ayikla(pdf_bytes: bytes):
    """CV PDF dosyasını metne dönüştürür (PyPDF2 Kullanarak)."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        metin = ""
        for page in pdf_reader.pages:
            sayfa_metni = page.extract_text()
            if sayfa_metni:
                metin += sayfa_metni + "\n"
        
        if not metin.strip():
            raise ValueError("PDF'den metin okunamadı.")
            
        return metin
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF Okuma Hatası: {str(e)}")

# 3. API Uç Noktaları (Endpoints)

@app.get("/")
async def root():
    return {
        "message": "Resumise AI Engine is Running!",
        "status": "online",
        "documentation": "/docs"
    }

@app.post("/analyze-link")
async def analyze_with_link(cv_file: UploadFile = File(...), job_url: str = Form(...)):
    """CV ve Link üzerinden Semantik Skorlama yapar."""
    pdf_content = await cv_file.read()
    cv_text = pdf_metin_ayikla(pdf_content)
    
    job_description = linkten_metin_cek(job_url)
    if job_description.startswith("Hata:"):
        return {"status": "error", "message": job_description}
    
    cv_vector = embed_model.encode(cv_text, convert_to_tensor=True)
    job_vector = embed_model.encode(job_description, convert_to_tensor=True)
    
    score = round(float(util.pytorch_cos_sim(cv_vector, job_vector)) * 100, 2)
    score = max(0, score)
    
    return {
        "status": "success",
        "data": {
            "score": score,
            "job_summary": job_description[:300] + "..."
        }
    }

@app.post("/get-advice")
async def get_ai_advice(cv_file: UploadFile = File(...), job_description: str = Form(...)):
    """Gemini Kariyer Koçu: Tavsiyeler sunar."""
    pdf_content = await cv_file.read()
    cv_text = pdf_metin_ayikla(pdf_content)
    
    prompt = f"""
    Sen profesyonel bir İK Direktörü ve Kariyer Koçusun. 
    Aday CV'si: {cv_text}
    İş İlanı: {job_description}
    
    Lütfen şunları Türkçe olarak analiz et:
    1. Adayın güçlü yanları.
    2. Eksik yetenekler (Skill Gap).
    3. CV için 3 kritik öneri.
    4. 2 adet mülakat sorusu ve cevabı.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return {
            "status": "success",
            "ai_advice": response.text
        }
    except Exception as e:
        return {"status": "error", "message": f"AI Hatası: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
