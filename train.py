"""
train.py — รัน python train.py เพื่อ train โมเดลใหม่
บันทึกโมเดลไปที่ models/tuned_trading_model.joblib โดยตรง
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_score
from sklearn.utils.class_weight import compute_sample_weight

from src.data_loader import get_processed_data

# ============================================================
# CONFIG — ปรับได้ตามต้องการ
# ============================================================
SYMBOL          = "XAUUSD"
INTERVAL        = "1m"
NUM_BARS        = 10000
FORWARD_BARS    = 3       # มองข้างหน้ากี่แท่ง
TARGET_POINTS   = 5       # ราคาต้องเคลื่อน >= $0.50
CONFIDENCE_THRESHOLD = 0.60

FEATURES = [
    'rsi_3', 'rsi_7', 'rsi_14',
    'stoch_k', 'stoch_d',
    'ema_fast_cross', 'ema_trend',
    'macd_diff', 'macd_line',
    'atr_3', 'atr_7', 'atr_ratio',
    'bb_width', 'bb_position',
    'candle_body', 'upper_wick', 'lower_wick', 'is_bullish', 'hl_range',
    'log_returns_lag_1', 'log_returns_lag_2', 'log_returns_lag_3',
    'ret_lag_1', 'ret_lag_2', 'ret_lag_3', 'ret_lag_4', 'ret_lag_5',
    'vol_ratio', 'dist_to_ema9', 'ema_cross',
]

# ============================================================
# 1. ดึงข้อมูล
# ============================================================
print(f"[1/4] ดึงข้อมูล {SYMBOL} {INTERVAL} จาก MT5...")
df = get_processed_data(SYMBOL, interval=INTERVAL, num_bars=NUM_BARS)

if df is None or len(df) == 0:
    print("❌ ดึงข้อมูลไม่ได้ หยุดการทำงาน")
    sys.exit(1)

print(f"      ได้ข้อมูล {len(df)} แถว | BUY: {(df.target==1).sum()} | SELL: {(df.target==0).sum()}")

# ============================================================
# 2. เตรียม X, y
# ============================================================
print("[2/4] เตรียม Features...")
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"      Features ที่ใช้: {len(FEATURES)} ตัว")

X = df[FEATURES]
y = df['target']

split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ============================================================
# 3. Train
# ============================================================
print("[3/4] Training XGBoost...")
sample_weights = compute_sample_weight('balanced', y_train)

model = XGBClassifier(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.5,
    eval_metric='logloss',
    random_state=42,
    early_stopping_rounds=50,
)

model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_test, y_test)],
    verbose=100
)

print(f"      Best iteration: {model.best_iteration}")

# ============================================================
# 4. วัดผล
# ============================================================
print("[4/4] วัดผล...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=['SELL (0)', 'BUY (1)']))

buy_prec  = precision_score(y_test, y_pred, pos_label=1)
sell_prec = precision_score(y_test, y_pred, pos_label=0)
print(f"🎯 BUY Precision:  {buy_prec*100:.1f}%")
print(f"🎯 SELL Precision: {sell_prec*100:.1f}%")

# Confidence filter
high_conf = (y_prob >= CONFIDENCE_THRESHOLD) | (y_prob <= 1 - CONFIDENCE_THRESHOLD)
if high_conf.sum() > 0:
    hc_prec = precision_score(y_test[high_conf], y_pred[high_conf], pos_label=1)
    print(f"✨ Confidence>{CONFIDENCE_THRESHOLD*100:.0f}% Precision: {hc_prec*100:.1f}% ({high_conf.sum()} สัญญาณ)")

# ============================================================
# 5. บันทึก
# ============================================================
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/tuned_trading_model.joblib')

config = {
    'features': FEATURES,
    'confidence_threshold': CONFIDENCE_THRESHOLD,
    'forward_bars': FORWARD_BARS,
    'target_points': TARGET_POINTS,
    'interval': INTERVAL,
}
joblib.dump(config, 'models/model_config.joblib')

print("\n✅ บันทึกโมเดลเสร็จแล้ว!")
print(f"   → models/tuned_trading_model.joblib")
print(f"   Features: {len(FEATURES)} ตัว | Confidence: {CONFIDENCE_THRESHOLD}")
print(f"\n⚠️  Re-Train ใหม่ทุก 2-3 วันเพราะ M1 เปลี่ยนเร็ว!")
