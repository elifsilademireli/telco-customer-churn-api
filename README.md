# 🚀 Uçtan Uca Müşteri Kaybı (Churn) Tahmin Sistemi ve REST API

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)

Bu proje, telekomünikasyon sektöründe kritik bir iş problemi olan **müşteri kaybını (churn)** önceden tahmin etmek amacıyla geliştirilmiş uçtan uca bir veri bilimi ve yazılım mühendisliği çözümüdür.

Proje kapsamında:
- Veri ön işleme
- Model geliştirme ve karşılaştırma
- En iyi modelin seçilmesi
- Modelin **FastAPI** ile servis edilmesi  

gibi tüm yaşam döngüsü eksiksiz şekilde ele alınmıştır.

---

## 🎯 İş Problemi ve Amaç

Telekom sektöründe yeni müşteri kazanmak, mevcut müşteriyi elde tutmaktan çok daha maliyetlidir.

Bu projenin amacı:
- Yüksek churn riski taşıyan müşterileri önceden tespit etmek  
- Bu tahminleri gerçek zamanlı kullanılabilir hale getirmek  
- CRM ve benzeri sistemlere entegre edilebilecek bir **API servisi** sunmaktır  

---

## 📊 Veri Seti ve EDA Özeti

Model eğitimi için **Kaggle Telco Customer Churn** veri seti kullanılmıştır.

- 👥 **Toplam müşteri:** 7,043  
- 🔢 **Toplam özellik:** 21  

### Veri Kategorileri

**1. Demografik Bilgiler**
- Cinsiyet  
- SeniorCitizen  
- Partner  
- Dependents  

**2. Hesap Bilgileri**
- Tenure (müşterilik süresi)  
- Contract (sözleşme türü)  
- PaymentMethod  
- MonthlyCharges  
- TotalCharges  

**3. Servis Kullanımı**
- InternetService  
- PhoneService  
- OnlineSecurity  
- TechSupport  
- StreamingTV / Movies  

---

## 🧠 Makine Öğrenmesi Pipeline Mimarisi

Projede **Scikit-Learn Pipeline** kullanılarak uçtan uca bir yapı kurulmuştur.

Bu yaklaşım:
- Veri sızıntısını (data leakage) önler  
- Production ortamında tutarlılığı artırır  

---

### 🔧 1. Veri Ön İşleme

- **Veri Temizliği**
  - `TotalCharges` içindeki boş string değerler → `NaN` olarak düzeltildi  

- **Özellik Seçimi**
  - `customerID` kaldırıldı  

- **Dönüşümler**
  - Sayısal veriler → `StandardScaler`  
  - Kategorik veriler → `OneHotEncoder(drop='first')`  

---

### 🤖 2. Model Geliştirme

Aşağıdaki modeller eğitilip karşılaştırılmıştır:

1. Logistic Regression  
2. Random Forest  
3. XGBoost  

---

### 🏆 Şampiyon Model

**Logistic Regression (~%80.45 Accuracy)**

📌 Neden?
- Veri setindeki ilişkiler büyük ölçüde doğrusal  
- Daha basit model → daha iyi genelleme  
- Overfitting riski daha düşük  

Model şu dosyaya kaydedilmiştir:

```bash
best_churn_model.pkl
📂 Proje Dizin Yapısı
📦 Telco-Customer-Churn
┣ 📂 data
┃ ┗ 📜 WA_Fn-UseC_-Telco-Customer-Churn.csv
┣ 📂 scripts
┃ ┣ 📜 preprocess.py
┃ ┗ 📜 evaluate.py
┣ 📜 train.py
┣ 📜 app.py
┣ 📜 requirements.txt
┣ 📜 Dockerfile
┗ 📜 README.md
⚙️ Kurulum ve Çalıştırma
🖥️ Yöntem 1: Lokal Ortam
1. Repo'yu klonla
git clone https://github.com/elifsilademireli/telco-customer-churn-api.git
cd telco-churn-api
2. Bağımlılıkları yükle
pip install -r requirements.txt
3. (Opsiyonel) Modeli eğit
python train.py
4. API’yi başlat
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Swagger UI:

http://localhost:8000/docs
🐳 Yöntem 2: Docker
docker build -t churn-prediction-api .
docker run -d -p 8000:8000 churn-prediction-api
📡 API Dokümantasyonu
🔹 Endpoint
POST /predict
📥 Request (JSON)
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 24,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.85,
  "TotalCharges": 2156.40
}
📤 Response (JSON)
{
  "Churn": "Yes",
  "Olasılık": "%72.14"
}
💡 Gelecek Geliştirmeler
🎨 Kullanıcı Arayüzü
Streamlit veya Gradio ile frontend geliştirme
🔍 Model Yorumlanabilirliği (XAI)
SHAP ile feature etkilerinin analiz edilmesi
API response içine açıklamalar eklenmesi
🔄 CI/CD Entegrasyonu
GitHub Actions ile otomatik deploy
Model güncellemelerinin otomasyonu
