# 🤖 Resumise AI Engine - Teknik Dokümantasyon

Bu doküman, Resumise projesinin **AI Engine (Python/FastAPI)** servisinin Spring Boot backend ile entegrasyonu için hazırlanmıştır.

## 🚀 Genel Bilgiler
- **Base URL:** `http://localhost:8000`
- **Etkileşimli Dokümantasyon (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Teknoloji:** FastAPI, Gemini 2.0 Flash, Sentence-Transformers (NLP).

---

## 🛠 API Uç Noktaları (Endpoints)

### 1. Sağlık Kontrolü (Health Check)
Servisin ayakta olup olmadığını kontrol etmek için kullanılır.
- **Method:** `GET`
- **Path:** `/`
- **Response:**
```json
{
  "message": "Resumise AI Engine is Running!",
  "status": "online",
  "documentation": "/docs"
}
```

---

### 2. CV ve Link Analizi (Semantik Skorlama)
Bir PDF CV ile web üzerindeki bir iş ilanı linkini karşılaştırıp benzerlik skoru üretir.
- **Method:** `POST`
- **Path:** `/analyze-link`
- **Request (Multipart/Form-Data):**
    - `cv_file`: PDF dosyası (binary)
    - `job_url`: String (İş ilanı URL'si)
- **Response:**
```json
{
  "status": "success",
  "data": {
    "score": 85.42,
    "job_summary": "İş ilanı özeti metni...",
    "analysis_type": "semantic-cosine-similarity"
  }
}
```

---

### 3. Kariyer Koçu (AI Tavsiyeleri)
Gemini 2.0 kullanarak CV ve iş tanımı arasındaki farkları analiz eder ve yapılandırılmış öneriler sunar.
- **Method:** `POST`
- **Path:** `/get-advice`
- **Request (Multipart/Form-Data):**
    - `cv_file`: PDF dosyası (binary)
    - `job_description`: String (İş ilanı metni)
- **Response:**
```json
{
  "status": "success",
  "data": {
    "strengths": "Adayın güçlü yanları...",
    "skill_gap": "Eksik yeteneklerin listesi...",
    "cv_suggestions": "3 kritik iyileştirme önerisi...",
    "interview_prep": "2 teknik mülakat sorusu ve cevabı..."
  }
}
```

---

## ⚠️ Hata Yönetimi (Error Handling)
Hata durumlarında API her zaman şu yapıda bir yanıt döner:
```json
{
  "status": "error",
  "message": "Hata detayı burada yer alır."
}
```
- **HTTP 400:** PDF okuma hatası veya eksik parametre.
- **HTTP 429:** AI model kota aşımı (Lütfen 30 sn sonra tekrar deneyin).
- **HTTP 500:** Sunucu taraflı beklenmedik hata.

---

## 📦 Kurulum (Geliştirici Notu)
Projeyi ayağa kaldırmak için:
1. `pip install -r requirements.txt`
2. `python main.py`

---
*Hazırlayan: Resumise AI Team*
