import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score

def prepare_train_test_data(df: pd.DataFrame, features: list, target: str, test_size: float = 0.2):
    """
    แบ่งข้อมูลเป็น Train และ Test แบบ Time-Series (ห้าม Shuffle)
    """
    X = df[features]
    y = df[target]
    
    # หาจุดตัด (Index) สำหรับแบ่งข้อมูล
    split_idx = int(len(df) * (1 - test_size))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train size: {len(X_train)} rows | Test size: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test

def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    เทรนโมเดลด้วย HistGradientBoosting (แฝดของ XGBoost แก้ปัญหาบน Mac)
    """
    print("Training Model...")
    
    model = HistGradientBoostingClassifier(
        max_iter=150,           # จำนวนต้นไม้ (เทียบเท่า n_estimators)
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    ประเมินความแม่นยำของโมเดล โดยเน้นที่ Precision
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # ความน่าจะเป็นที่จะเป็น 1 (ราคาขึ้น)
    
    acc = accuracy_score(y_test, y_pred)
    # Precision คือ: ทายว่าขึ้น 100 ครั้ง มันขึ้นจริงๆ กี่ครั้ง (สำคัญมากสำหรับบอทเทรด)
    prec = precision_score(y_test, y_pred, zero_division=0)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy : {acc:.4f} (ความแม่นยำรวม)")
    print(f"Precision: {prec:.4f} (ความแม่นยำเมื่อบอกให้ ซื้อ)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # ส่งคืนค่า Prediction และ Probability กลับไปใช้งานต่อ
    return y_pred, y_prob