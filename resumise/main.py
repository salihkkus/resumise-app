import io
import requests
import google.generativeai as genai
import PyPDF2
from bs4 import BeautifulSoup
from fastapi import FastAPI, UploadFile, File, Form
from sentence_transformers import SentenceTransformer, util

# 1. Uygulama ve Model Başlatma
app = FastAPI(title="Resumise AI Engine")

# NLP Modeli (Anlamsal analiz için)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Gemini Yapay Zeka Ayarları
GEMINI_API_KEY = "AIzaSyAEXYL5rCHkuTXponb9eQ_RxU1k9vUF6IM" # Sunum sonrası bunu gizli tutmalısın
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# 2. Yardımcı Fonksiyonlar
def linkten_metin_cek(url: str):
    """Verilen linkteki web sayfasından metin içeriğini temizleyerek çeker."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        yanit = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(yanit.text, 'html.parser')
        
        # Gereksiz etiketleri (script, style) temizle
        for script in soup(["script", "style"]):
            script.extract()
            
        metin = soup.get_text()
        # Satır aralarındaki gereksiz boşlukları temizle
        satirlar = (line.strip() for line in metin.splitlines())
        temiz_metin = '\n'.join(chunk for chunk in satirlar if chunk)
        return temiz_metin
    except Exception as e:
        return f"Link okuma hatası: {e}"

def pdf_metin_ayikla(pdf_bytes: bytes):
    """Yüklenen PDF dosyasının içeriğini metne dönüştürür."""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    metin = ""
    for page in pdf_reader.pages:
        metin += page.extract_text()
    return metin

# 3. API Uç Noktaları (Endpoints)

@app.post("/analyze-link")
async def analyze_with_link(cv_file: UploadFile = File(...), job_url: str = Form(...)):
    """CV ve Link üzerinden eşleştirme skoru hesaplar."""
    # CV Metnini al
    pdf_content = await cv_file.read()
    cv_text = pdf_metin_ayikla(pdf_content)
    
    # İş İlanı Metnini linkten çek
    job_description = linkten_metin_cek(job_url)
    
    # NLP Skorlama (Cosine Similarity)
    cv_vector = model.encode(cv_text, convert_to_tensor=True)
    job_vector = model.encode(job_description, convert_to_tensor=True)
    score = round(float(util.pytorch_cos_sim(cv_vector, job_vector)) * 100, 2)
    
    return {
        "status": "success",
        "score": score,
        "job_preview": job_description[:200] + "..."
    }

@app.post("/get-advice")
async def get_ai_advice(cv_file: UploadFile = File(...), job_description: str = Form(...)):
    """Gemini API kullanarak kariyer tavsiyesi ve mülakat soruları üretir."""
    pdf_content = await cv_file.read()
    cv_text = pdf_metin_ayikla(pdf_content)
    
    prompt = f"""
    Sen profesyonel bir İK uzmanı ve kariyer koçusun. 
    Aşağıdaki CV'yi ve iş ilanını analiz et:
    
    CV: {cv_text}
    İş İlanı: {job_description}
    
    Senden şunları istiyorum:
    1. Bu aday bu işe neden uygun veya neden değil? (Kısa özet)
    2. CV'de değiştirilmesi veya eklenmesi gereken 3 kritik şey nedir?
    3. Bu iş görüşmesinde adaya sorulabilecek 3 adet teknik mülakat sorusu ve ideal cevapları nedir?
    
    Lütfen yanıtını profesyonel ve motive edici bir dille, Türkçe olarak ver.
    """
    
    response = gemini_model.generate_content(prompt)
    
    return {
        "status": "success",
        "ai_advice": response.text
    }

if __name__ == "__main__":
    import uvicorn
    # Uygulamayı 8000 portunda başlat
    uvicorn.run(app, host="0.0.0.0", port=8000)