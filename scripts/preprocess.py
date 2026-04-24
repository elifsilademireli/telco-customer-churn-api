# scripts/preprocess.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_and_clean_data(filepath="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """Veriyi yükler, temizler ve hedef değişkeni ayarlar."""
    df = pd.read_csv(filepath)
    
    # 1. TotalCharges kolonunda gizli boşluklar (' ') var. Bunları NaN yapıp siliyoruz.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    
    # 2. customerID modelin tahmin gücünü etkilemez, ezber yaptırır. Siliyoruz.
    df = df.drop('customerID', axis=1)
    
    # 3. Churn (Hedef Değişken) Yes/No formatında, bunu 1/0 formatına çeviriyoruz.
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    return df

def get_features_and_target(df):
    """Veriyi X (özellikler) ve y (hedef) olarak ayırır."""
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    return X, y

def get_preprocessor(X):
    """
    Sayısal kolonları ölçeklendiren (StandardScaler) ve 
    Kategorik kolonları 1-0 matrisine çeviren (OneHotEncoder) pipeline'ı döner.
    """
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    # Sayısal olmayan diğer tüm kolonları kategorik olarak kabul et
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            # drop='first' -> Dummy variable trap (çoklu doğrusallık) problemini önler
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features) 
        ])
    
    return preprocessor