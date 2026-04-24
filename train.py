# train.py
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier # Ekstra Model Puanı İçin (pip install xgboost)

# Kendi yazdığımız modülleri (scriptleri) içeri aktarıyoruz
from scripts.preprocess import load_and_clean_data, get_features_and_target, get_preprocessor
from scripts.evaluate import compare_and_select_best

def main():
    print("1. Veri yükleniyor ve temizleniyor...")
    df = load_and_clean_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    print("2. Veri Eğitim ve Test olarak ayrılıyor...")
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("3. Ön işleme (Preprocessing) kuralları hazırlanıyor...")
    preprocessor = get_preprocessor(X)
    
    print("4. Modeller oluşturuluyor...")
    # Pipeline sayesinde veri önce preprocessor'dan geçer, sonra modele girer.
    models_to_test = {
        "Logistic Regression": Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))]),
        "Random Forest": Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))]),
        "XGBoost": Pipeline([('preprocessor', preprocessor), ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))])
    }
    
    print("5. Modeller yarışıyor...")
    # Evaluate modülümüzdeki fonksiyonu çağırıyoruz
    best_pipeline = compare_and_select_best(models_to_test, X_train, X_test, y_train, y_test)
    
    print("6. En iyi model kaydediliyor...")
    joblib.dump(best_pipeline, 'best_churn_model.pkl')
    print("İşlem Başarıyla Tamamlandı! Artık API'yi (app.py) çalıştırabilirsiniz.")

if __name__ == "__main__":
    main()