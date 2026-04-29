# 🚀 Uçtan Uca Müşteri Kaybı (Churn) Tahmin Sistemi ve REST API

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)

Bu proje, telekomünikasyon sektöründe kritik bir iş problemi olan **Müşteri Kaybını (Churn)** önceden tespit etmek amacıyla geliştirilmiş uçtan uca bir veri bilimi ve yazılım mühendisliği çözümüdür. Çalışma; veri ön işleme süreçlerinden başlayarak, makine öğrenmesi modellerinin karşılaştırılmasına ve nihayetinde şampiyon modelin **FastAPI** ile bir web servisi (API) olarak canlıya alınmasına kadar olan tüm yaşam döngüsünü kapsar.

Model başarıyla eğitilmiş ve şu dosyaya kaydedilmiştir: `best_churn_model.pkl`

---

## 🎯 İş Problemi ve Projenin Amacı

Telekom şirketleri için yeni bir müşteri kazanmanın maliyeti, mevcut müşteriyi elde tutmaktan çok daha yüksektir. Bu sistemin temel amacı:
- Şirketten ayrılma riski yüksek olan müşterileri davranışsal ve demografik verilerinden tespit etmek.
- Bu tespitleri dış uygulamaların (CRM sistemleri, mobil uygulamalar vb.) anında kullanabileceği, ölçeklenebilir ve düşük gecikmeli bir **API servisi** üzerinden sunmaktır.

---

## 📊 Veri Seti ve Keşifsel Analiz (EDA) Özeti

Modelin eğitiminde **Kaggle Telco Customer Churn** veri seti kullanılmıştır. Veri seti 7.043 müşteri ve 21 özellikten oluşmaktadır. Veriler üç ana grupta toplanır:

1. **Demografik Veriler:** Cinsiyet, yaşlılık durumu, partner ve bakmakla yükümlü olunan kişi durumu.
2. **Hesap Bilgileri:** Müşterilik süresi (tenure), sözleşme türü, ödeme yöntemi, aylık ve toplam fatura tutarları.
3. **Servis Kullanımı:** İnternet türü, telefon servisi, teknik destek ve güvenlik paketleri abonelik durumları.

---

## 🧠 Makine Öğrenmesi Mimarisi (Pipeline)

Projede, verinin ham halden tahmin üretecek duruma gelmesi için katı bir **Scikit-Learn Pipeline** mimarisi kurulmuştur. Bu yapı, veri sızıntısını (data leakage) engeller ve canlı ortamda (production) yüksek tutarlılık sağlar.

### 1. Veri Ön İşleme (Preprocessing) Stratejisi
* **Veri Temizliği:** `TotalCharges` kolonunda tespit edilen görünmez boşluk karakterleri (' ') `NaN` değerlere dönüştürülmüş ve eksik veriler temizlenmiştir.
* **Özellik Seçimi:** Modelin ezber (overfitting) yapmasını önlemek adına, hiçbir tahmin gücü olmayan `customerID` değişkeni veri setinden çıkarılmıştır.
* **Dönüşümler (Transformations):**
  * **Sayısal Değişkenler:** `tenure`, `MonthlyCharges` ve `TotalCharges` özellikleri `StandardScaler` ile ölçeklendirilmiştir.
  * **Kategorik Değişkenler:** Çoklu doğrusallık (dummy variable trap) problemini önlemek adına `OneHotEncoder(drop='first')` kullanılarak makine diline çevrilmiştir.

### 2. Model Geliştirme ve Karşılaştırma
Sistem üç farklı algoritmayı eğitmiş ve test verisi üzerinde birbiriyle yarıştırtmıştır:
1. `Logistic Regression`
2. `Random Forest Classifier`
3. `XGBoost Classifier`

> **🏆 Şampiyon Model: Logistic Regression (~%80.45 Accuracy)**
> Telekom churn verisi gibi doğrusal ilişkilerin (örneğin: sözleşme süresi arttıkça müşteri kaybı azalır) belirgin olduğu durumlarda, karmaşık ağaç algoritmaları yerine iyi ölçeklendirilmiş bir Lojistik Regresyon modelinin en iyi performansı gösterdiği tespit edilmiştir.

---

## 📂 Proje Dizin Yapısı

Mimari, kodun okunabilirliğini ve sürdürülebilirliğini sağlamak adına modüler olarak tasarlanmıştır:

```text
📦 Telco-Customer-Churn
 ┣ 📂 data
 ┃ ┗ 📜 WA_Fn-UseC_-Telco-Customer-Churn.csv   # Orijinal Veri Seti
 ┣ 📂 scripts
 ┃ ┣ 📜 preprocess.py                          # Veri temizleme modülü
 ┃ ┗ 📜 evaluate.py                            # Metrik ve model seçim modülü
 ┣ 📜 train.py                                 # Pipeline orkestratörü ve eğitim scripti
 ┣ 📜 app.py                                   # FastAPI sunucusu
 ┣ 📜 requirements.txt                         # Bağımlılıklar
 ┣ 📜 Dockerfile                               # Konteyner imaj dosyası
 ┗ 📜 README.md                                # Proje dokümantasyonu





## 📊 Model Performans Karşılaştırması

Model seçimi **Doğruluk Skoru (Accuracy)** metriğine göre yapılmıştır. Yapılan testler sonucunda en yüksek skora ulaşan model şampiyon olarak belirlenmiş ve API'ye entegre edilmiştir.

| Model | Doğruluk Skoru (Accuracy) |
| :--- | :--- |
| 🏆 **Logistic Regression** | **0.8045** |
| Random Forest | 0.7839 |
| XGBoost | 0.7783 |

<details>
<summary><b>1. Logistic Regression Sonuçları (Detayları Görmek İçin Tıklayın)</b></summary>

```text
Doğruluk Skoru (Accuracy) : 0.8045

              precision    recall  f1-score   support

           0       0.85      0.89      0.87      1033
           1       0.65      0.57      0.61       374

    accuracy                           0.80      1407
   macro avg       0.75      0.73      0.74      1407
weighted avg       0.80      0.80      0.80      1407



### Diğer Modellerin Detaylı Sonuçları

```text
--------------- Random Forest Sonuçları ---------------
Doğruluk Skoru (Accuracy) : 0.7839
Detaylı Sınıflandırma Raporu:
              precision    recall  f1-score   support

           0       0.83      0.89      0.86      1033
           1       0.62      0.49      0.54       374

    accuracy                           0.78      1407
   macro avg       0.72      0.69      0.70      1407
weighted avg       0.77      0.78      0.78      1407


--------------- XGBoost Sonuçları ---------------
Doğruluk Skoru (Accuracy) : 0.7783
Detaylı Sınıflandırma Raporu:
              precision    recall  f1-score   support

           0       0.84      0.86      0.85      1033
           1       0.59      0.55      0.57       374

    accuracy                           0.78      1407
   macro avg       0.72      0.70      0.71      1407
weighted avg       0.77      0.78      0.78      1407













## ⚙️ Kurulum ve Çalıştırma
Sistemi kendi bilgisayarınızda veya sunucunuzda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

🖥️ Yöntem 1: Lokal Ortam
1. Repo'yu klonla:

Bash
git clone [https://github.com/elifsilademireli/telco-customer-churn-api.git](https://github.com/elifsilademireli/telco-customer-churn-api.git)
cd telco-churn-api
2. Bağımlılıkları yükle:

Bash
pip install -r requirements.txt
3. (Opsiyonel) Modeli eğit:

Bash
python train.py
4. API'yi başlat:

Bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
Swagger UI: API'yi interaktif olarak test etmek için tarayıcınızdan http://localhost:8000/docs adresine gidebilirsiniz.

🐳 Yöntem 2: Docker
Projeyi kendi ortam değişkenlerinizden bağımsız, izole bir şekilde çalıştırmak isterseniz:

Bash
docker build -t churn-prediction-api .
docker run -d -p 8000:8000 churn-prediction-api
📡 API Dokümantasyonu
API, POST metoduyla çalışan ve Pydantic ile sıkı veri doğrulaması (data validation) yapan bir uç noktaya sahiptir.

🔹 Endpoint: POST /predict

📥 Request (JSON)
Müşteri özelliklerini alır ve makine öğrenmesi modeli üzerinden churn (kayıp) tahminini olasılık değeriyle birlikte döner.

JSON
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

JSON
{
  "Churn": "Yes",
  "Olasılık": "%72.14"
}
💡 Gelecek Geliştirmeler
🎨 Kullanıcı Arayüzü: Streamlit veya Gradio ile son kullanıcıya (kod bilmeyen yöneticilere) hitap eden frontend geliştirilmesi.

🔍 Model Yorumlanabilirliği (XAI): SHAP (SHapley Additive exPlanations) ile özellik (feature) etkilerinin analiz edilmesi ve API response içerisine tahminin nedenlerine dair açıklamalar eklenmesi.

🔄 CI/CD Entegrasyonu: GitHub Actions ile otomatik dağıtım (deploy) süreçlerinin ve model güncellemelerinin otomatikleştirilmesi.
