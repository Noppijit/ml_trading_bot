import pandas as pd
import numpy as np
import ta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import EMAIndicator, MACD
import MetaTrader5 as mt5

# ============================================================
# 1. Features สำหรับ Scalping M1 โดยเฉพาะ
# ============================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[Feature] กำลังสร้าง Indicators สำหรับ Scalping M1...")
    data = df.copy()
    data.columns = [str(c).lower() for c in data.columns]

    # --- Price Returns ---
    data['returns']     = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))

    # --- RSI สั้นมาก (M1 ต้องตอบสนองเร็ว) ---
    data['rsi_3']  = RSIIndicator(close=data['close'], window=3).rsi()
    data['rsi_7']  = RSIIndicator(close=data['close'], window=7).rsi()
    data['rsi_14'] = RSIIndicator(close=data['close'], window=14).rsi()

    # --- Stochastic (ดี สำหรับ overbought/oversold M1) ---
    stoch = StochasticOscillator(
        high=data['high'], low=data['low'], close=data['close'],
        window=5, smooth_window=3
    )
    data['stoch_k'] = stoch.stoch()
    data['stoch_d'] = stoch.stoch_signal()

    # --- MACD สั้น ---
    macd = MACD(close=data['close'], window_slow=12, window_fast=5, window_sign=3)
    data['macd_diff'] = macd.macd_diff()
    data['macd_line'] = macd.macd()

    # --- ATR สั้น (volatility filter) ---
    data['atr_3'] = AverageTrueRange(
        high=data['high'], low=data['low'], close=data['close'], window=3
    ).average_true_range()
    data['atr_7'] = AverageTrueRange(
        high=data['high'], low=data['low'], close=data['close'], window=7
    ).average_true_range()
    data['atr_ratio'] = data['atr_3'] / data['atr_7']  # > 1 = ตลาดร้อน

    # --- Bollinger Bands สั้น ---
    bb = BollingerBands(close=data['close'], window=10, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    data['bb_width']    = (bb_upper - bb_lower) / data['close']
    data['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # --- EMA สั้น สำหรับ trend M1 ---
    ema3  = EMAIndicator(close=data['close'], window=3).ema_indicator()
    ema7  = EMAIndicator(close=data['close'], window=7).ema_indicator()
    ema21 = EMAIndicator(close=data['close'], window=21).ema_indicator()
    data['ema_fast_cross'] = (ema3 - ema7) / data['close']   # momentum สั้น
    data['ema_trend']      = (ema7 - ema21) / data['close']  # trend กลาง
    data['dist_to_ema9']   = (data['close'] - ema7) / ema7
    data['ema_cross']      = ema3 - ema7

    # --- Price Action (สำคัญมากสำหรับ Scalping) ---
    data['candle_body']  = abs(data['close'] - data['open']) / (data['atr_7'] + 1e-9)
    data['upper_wick']   = (data['high'] - data[['close','open']].max(axis=1)) / (data['atr_7'] + 1e-9)
    data['lower_wick']   = (data[['close','open']].min(axis=1) - data['low']) / (data['atr_7'] + 1e-9)
    data['is_bullish']   = (data['close'] > data['open']).astype(int)
    data['hl_range']     = (data['high'] - data['low']) / (data['atr_7'] + 1e-9)

    # --- Momentum Lag 1-5 แท่ง ---
    for i in range(1, 6):
        data[f'log_returns_lag_{i}'] = data['log_returns'].shift(i)
        data[f'ret_lag_{i}']         = data['returns'].shift(i)

    # --- Volume Spike ---
    if 'volume' in data.columns and data['volume'].sum() > 0:
        data['vol_ratio'] = data['volume'] / (data['volume'].rolling(20).mean() + 1e-9)
    else:
        data['vol_ratio'] = 1.0

    # ============================================================
    # TARGET สำหรับ Scalping M1
    # ราคาต้องเคลื่อน >= TARGET_POINTS ใน FORWARD_BARS แท่งข้างหน้า
    # ============================================================
    FORWARD_BARS  = 3     # มองข้างหน้า 3 แท่ง M1
    TARGET_POINTS = 5     # ราคาต้องเคลื่อน >= $0.50

    future_close = data['close'].shift(-FORWARD_BARS)
    data['target'] = np.where(
        future_close >= data['close'] + TARGET_POINTS * 0.01, 1,   # BUY
        np.where(
            future_close <= data['close'] - TARGET_POINTS * 0.01, 0,  # SELL
            np.nan  # NEUTRAL — ตัดทิ้ง
        )
    )

    # ลบ NaN
    before = len(data)
    data   = data.dropna()
    after  = len(data)
    print(f"[Feature] ลบ NaN ออก {before - after} แถว → เหลือ {after} แถว")
    print(f"[Feature] BUY: {(data.target==1).sum()} | SELL: {(data.target==0).sum()}")

    return data


# ============================================================
# 2. Pipeline หลัก
# ============================================================
def get_processed_data(symbol: str = "XAUUSD", interval: str = "1m", num_bars: int = 10000):
    tf_map = {
        "1m":  mt5.TIMEFRAME_M1,
        "5m":  mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "1h":  mt5.TIMEFRAME_H1,
        "1d":  mt5.TIMEFRAME_D1
    }
    mt5_tf = tf_map.get(interval, mt5.TIMEFRAME_M1)

    if not mt5.initialize():
        print(f"❌ เชื่อมต่อ MT5 ไม่สำเร็จ (Error: {mt5.last_error()})")
        return None

    print(f"[Data] กำลังดึง {symbol} จาก MT5 | interval={interval} | แท่งย้อนหลัง={num_bars} ...")

    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, num_bars)
    if rates is None or len(rates) == 0:
        print(f"❌ ไม่พบข้อมูล {symbol}")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    print(f"[Data] ดึงข้อมูลสำเร็จ → {len(df)} แท่งเทียน")

    return build_features(df)