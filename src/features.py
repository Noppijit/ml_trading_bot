import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cross_decomposition import CCA

def calculate_rdc(x, y, f=np.sin, k=20, s=1/6., n=1, random_state=42):
    """
    คำนวณค่า Randomized Dependence Coefficient (RDC) ระหว่าง x และ y
    
    Parameters:
    - x, y: array-like (1D หรือ 2D) ข้อมูลฟีเจอร์และ Target
    - f: Non-linear function (ค่าเริ่มต้นคือ np.sin)
    - k: จำนวน Random features ที่ต้องการจำลอง (มิติที่เพิ่มขึ้น)
    - s: Scale ของการสุ่ม
    - n: จำนวน Components สำหรับ CCA
    """
    np.random.seed(random_state)
    
    # ทำให้แน่ใจว่าเป็น 2D Array
    if x.ndim == 1: x = x.reshape(-1, 1)
    if y.ndim == 1: y = y.reshape(-1, 1)
    
    # 1. Copula Transformation (Ranking ข้อมูล)
    cx = np.column_stack([rankdata(xc, method='ordinal') for xc in x.T]) / x.shape[0]
    cy = np.column_stack([rankdata(yc, method='ordinal') for yc in y.T]) / y.shape[0]
    
    # เตรียมเพิ่มมิติ (เติม bias term ด้วยเลข 1)
    O = np.ones(cx.shape[0])
    X_aug = np.column_stack([cx, O])
    Y_aug = np.column_stack([cy, O])
    
    # 2. Random Non-linear Projections
    Wx = s / X_aug.shape[1] * np.random.randn(X_aug.shape[1], k)
    Wy = s / Y_aug.shape[1] * np.random.randn(Y_aug.shape[1], k)
    
    fx = f(np.dot(X_aug, Wx))
    fy = f(np.dot(Y_aug, Wy))
    
    # 3. Canonical Correlation Analysis (CCA)
    cca = CCA(n_components=n)
    try:
        cca.fit(fx, fy)
        xc, yc = cca.transform(fx, fy)
        
        # รีเทิร์นค่า Correlation ที่สูงที่สุด
        corr = np.corrcoef(xc.T, yc.T)[0, 1]
        return abs(corr)
    except Exception as e:
        # กรณีข้อมูลมีปัญหาหรือ variance เป็น 0
        return 0.0

def select_features_with_rdc(df, feature_cols, target_col, threshold=0.1):
    """
    วนลูปหาค่า RDC ของทุกฟีเจอร์เทียบกับ Target และคัดเฉพาะตัวที่ผ่านเกณฑ์
    """
    print("Calculating RDC for feature selection...")
    y = df[target_col].values
    
    rdc_scores = {}
    for col in feature_cols:
        x = df[col].values
        # ข้ามฟีเจอร์ที่มีค่าคงที่ (ไม่มี Variance)
        if np.std(x) == 0:
            rdc_scores[col] = 0.0
            continue
            
        score = calculate_rdc(x, y)
        rdc_scores[col] = score
        
    # แปลงเป็น DataFrame ให้ดูง่ายๆ
    score_df = pd.DataFrame(list(rdc_scores.items()), columns=['Feature', 'RDC_Score'])
    score_df = score_df.sort_values(by='RDC_Score', ascending=False).reset_index(drop=True)
    
    print("\n--- Feature Importance (RDC) ---")
    print(score_df)
    
    # คัดฟีเจอร์ที่คะแนนผ่าน Threshold
    selected = score_df[score_df['RDC_Score'] >= threshold]['Feature'].tolist()
    print(f"\nSelected {len(selected)} features passing threshold >= {threshold}")
    
    return selected