import pandas as pd
import numpy as np
import ta
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import EMAIndicator, MACD
import MetaTrader5 as mt5

# ============================================================
# 1. สร้าง Feature สำหรับ Scalping (คำนวณ Indicator ครบทุกตัว)
# ============================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[Feature] กำลังสร้าง Indicators สำหรับ Scalping ...")
    data = df.copy()

    # ปรับชื่อคอลัมน์ให้เป็นตัวพิมพ์เล็กทั้งหมดเพื่อป้องกันปัญหา
    data.columns = [str(c).lower() for c in data.columns]

    # --- Price Returns ---
    data['returns']     = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))

    # --- Momentum ---
    data['rsi_7']    = RSIIndicator(close=data['close'], window=7).rsi()
    macd             = MACD(close=data['close'])
    data['macd_diff'] = macd.macd_diff()

    # --- Volatility ---
    data['atr_5'] = AverageTrueRange(
        high=data['high'], low=data['low'], close=data['close'], window=5
    ).average_true_range()

    bb = BollingerBands(close=data['close'], window=10, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    data['bb_width']    = (bb_upper - bb_lower) / data['close']
    data['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)  # 0=ล่าง, 1=บน

    # --- Trend ---
    ema_9  = EMAIndicator(close=data['close'], window=9).ema_indicator()
    ema_21 = EMAIndicator(close=data['close'], window=21).ema_indicator()
    data['dist_to_ema9'] = (data['close'] - ema_9) / ema_9
    data['ema_cross']    = ema_9 - ema_21  # บวก = EMA9 เหนือ EMA21 = Uptrend

    # --- Lag Features (บริบทย้อนหลัง 3 แท่ง) ---
    for i in range(1, 4):
        data[f'log_returns_lag_{i}'] = data['log_returns'].shift(i)

    # --- Target (สำหรับ Train โมเดล) ---
    data['target'] = np.where(data['returns'].shift(-1) > 0, 1, 0)

    # ลบแถว NaN ที่เกิดจาก Indicator และ Shift
    before = len(data)
    data   = data.dropna()
    print(f"[Feature] ลบ NaN ออก {before - len(data)} แถว → เหลือ {len(data)} แถว")

    return data


# ============================================================
# 2. Pipeline หลัก — ดึงข้อมูลจาก MT5 แล้วส่งไปทำ Features
# ============================================================
def get_processed_data(symbol: str = "XAUUSD", interval: str = "5m", num_bars: int = 5000):
    """
    ดึงข้อมูลจาก MT5 และสร้าง Features สำหรับเทรนโมเดล หรือบอทเทรด
    """
    
    # 1. แมปชื่อ Timeframe ให้ตรงกับคำสั่งของ MT5
    tf_map = {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "1h": mt5.TIMEFRAME_H1,
        "1d": mt5.TIMEFRAME_D1
    }
    mt5_tf = tf_map.get(interval, mt5.TIMEFRAME_M5)

    # 2. เชื่อมต่อโปรแกรม MT5 (ต้องเปิดโปรแกรม MT5 ล็อกอินทิ้งไว้ด้วย)
    if not mt5.initialize():
        print(f"❌ เชื่อมต่อ MT5 ไม่สำเร็จ (Error: {mt5.last_error()})")
        return None

    print(f"[Data] กำลังดึง {symbol} จาก MT5 | interval={interval} | แท่งย้อนหลัง={num_bars} ...")
    
    # 3. ดึงข้อมูลแท่งเทียน
    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, num_bars)
    if rates is None or len(rates) == 0:
        print(f"❌ ไม่พบข้อมูลสำหรับ {symbol} กรุณาเช็กว่าใน MT5 มีคู่เงินนี้หรือไม่")
        return None

    # 4. แปลงข้อมูลเป็น DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # 5. เปลี่ยนชื่อคอลัมน์ tick_volume ให้สั้นลงเป็น volume
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    print(f"[Data] ดึงข้อมูลสำเร็จ → {len(df)} แท่งเทียน")

    # 6. ส่ง DataFrame ไปคำนวณ Indicator ครบชุดผ่านฟังก์ชัน build_features
    df_featured = build_features(df)
    
    return df_featured