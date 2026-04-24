# scripts/evaluate.py
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def evaluate_model(model_name, y_true, y_pred):
    """Tek bir modelin performansını ölçer ve ekrana yazdırır."""
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{'-'*15} {model_name} Sonuçları {'-'*15}")
    print(f"Doğruluk Skoru (Accuracy) : {accuracy:.4f}")
    print("Detaylı Sınıflandırma Raporu:")
    print(classification_report(y_true, y_pred))
    return accuracy

def compare_and_select_best(models_dict, X_train, X_test, y_train, y_test):
    """
    Sözlük olarak verilen modelleri eğitir, test eder 
    ve doğruluk skoru en yüksek olan modeli döner.
    """
    best_acc = 0
    best_model = None
    best_model_name = ""
    
    for name, pipeline in models_dict.items():
        # Modeli Eğit
        pipeline.fit(X_train, y_train)
        
        # Test verisi ile tahmin yap
        y_pred = pipeline.predict(X_test)
        
        # Modeli Değerlendir
        acc = evaluate_model(name, y_test, y_pred)
        
        # En iyiyi bul
        if acc > best_acc:
            best_acc = acc
            best_model = pipeline
            best_model_name = name
            
    print("="*50)
    print(f"🌟 Şampiyon Model: {best_model_name} (Skor: {best_acc:.4f}) 🌟")
    print("="*50)
    
    return best_model