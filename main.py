import os
import sys
import joblib
import pandas as pd
from src.executor import execute_mt5_trade
import schedule
import time
from datetime import datetime, timedelta
import MetaTrader5 as mt5

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import get_processed_data
from src.risk_manager import calculate_position_size

# ===================================================
# ฟังก์ชันเช็กลิมิตออเดอร์ (สูงสุด 8 ไม้)
# ===================================================
def check_order_limit(symbol: str, max_orders: int = 8) -> bool:
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        print(f"❌ ดึงข้อมูลออเดอร์ไม่สำเร็จ (Error: {mt5.last_error()})")
        return True
    current_orders = len(positions)
    print(f"📊 ปัจจุบันมีออเดอร์ {symbol} เปิดอยู่: {current_orders}/{max_orders} ไม้")
    if current_orders >= max_orders:
        return True
    return False

def main():
    print("=== 🚀 เริ่มต้นการทำงาน ML Trading Bot ===")

    # -----------------------------------
    # 1. ตั้งค่าระบบ (Config)
    # -----------------------------------
    SYMBOL = "XAUUSD"
    ACCOUNT_BALANCE = 1000.00
    RISK_PCT = 0.05

    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

    # -----------------------------------
    # 2. เตรียมข้อมูล
    # -----------------------------------
    print(f"\n[1/4] ดึงข้อมูล {SYMBOL} ล่าสุดจากตลาด...")
    try:
        df = get_processed_data(SYMBOL, interval='5m', num_bars=5000)
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการดึงข้อมูล: {e}")
        return

    # -----------------------------------
    # 3. โหลดโมเดล AI
    # -----------------------------------
    print("\n[2/4] โหลดโมเดล AI ที่ปรับจูนไว้...")
    model_path = os.path.join("models", "tuned_trading_model.joblib")

    if not os.path.exists(model_path):
        print(f"❌ ไม่พบไฟล์โมเดลที่: {model_path}")
        return

    model = joblib.load(model_path)
    features = model.feature_names_in_.tolist()

    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        print(f"❌ ข้อมูลขาดฟีเจอร์เหล่านี้: {missing_cols}")
        return

    # -----------------------------------
    # 4. ทำนายสัญญาณ
    # -----------------------------------
    print("\n[3/4] วิเคราะห์ข้อมูลแท่งเทียนล่าสุด...")

    latest_data = df.iloc[-1:]
    X_latest = latest_data[features]
    signal = model.predict(X_latest)[0]  # คาดว่า return: 1=BUY, 0=SELL (หรือ -1=SELL)
    prob = model.predict_proba(X_latest)[0][1]

    print(f"   🔍 สัญญาณที่ได้: {signal} | ความมั่นใจ: {prob:.2%}")

    # -----------------------------------
    # 5. ดึงราคาและคำนวณ SL/TP แบบ Point
    # -----------------------------------
    tick = mt5.symbol_info_tick(SYMBOL)
    symbol_info = mt5.symbol_info(SYMBOL)

    if tick is None or symbol_info is None:
        print("❌ ดึงข้อมูลราคาจาก MT5 ไม่ได้")
        return

    current_price = tick.ask
    current_price_bid = tick.bid
    symbol_point = symbol_info.point

    # ✅ ตั้งระยะ SL/TP เป็น Point (ปรับได้ตามต้องการ)
    sl_points = 150    # SL = 350 points (~3.50 USD สำหรับทอง)
    tp_points = 400   # TP = 1000 points (~10.00 USD สำหรับทอง)

    sl_dist = sl_points * symbol_point
    tp_dist = tp_points * symbol_point

    print(f"\n[4/4] 🎯 แผนการเทรดวันนี้ (Action Plan)")
    print(f"   - Symbol Point: {symbol_point}")
    print(f"   - SL Distance: {sl_dist:.2f} | TP Distance: {tp_dist:.2f}")

    # -----------------------------------
    # 6. เช็กลิมิตออเดอร์ก่อนเปิดไม้ใหม่
    # -----------------------------------
    if check_order_limit(SYMBOL, max_orders=8):
        print("⚠️ ออเดอร์เต็มลิมิต 8 ไม้แล้ว รอปิดก่อนครับ")
        return

    # -----------------------------------
    # 7. ตัดสินใจและส่งคำสั่ง
    # -----------------------------------
    # ⚠️ เช็กว่าโมเดลของคุณ return อะไร: 1/0 หรือ 1/-1
    # ถ้า return 1=BUY, 0=SELL → ใช้ signal == 0
    # ถ้า return 1=BUY, -1=SELL → ใช้ signal == -1

    if signal == 1:
        sl_price = current_price - sl_dist
        tp_price = current_price + tp_dist

        print(f"🟢 สัญญาณ: BUY")
        print(f"   - ราคาเข้า: {current_price:.2f}")
        print(f"   - TP: {tp_price:.2f} | SL: {sl_price:.2f}")

        execute_mt5_trade(
            symbol=SYMBOL,
            order_type=mt5.ORDER_TYPE_BUY,
            volume=0.05,
            price=current_price,
            sl=sl_price,
            tp=tp_price
        )

    elif signal == 0 or signal == -1:  # รองรับทั้ง 2 แบบ
        sl_price = current_price_bid + sl_dist
        tp_price = current_price_bid - tp_dist

        print(f"🔴 สัญญาณ: SELL")
        print(f"   - ราคาเข้า: {current_price_bid:.2f}")
        print(f"   - TP: {tp_price:.2f} | SL: {sl_price:.2f}")

        execute_mt5_trade(
            symbol=SYMBOL,
            order_type=mt5.ORDER_TYPE_SELL,
            volume=0.05,
            price=current_price_bid,
            sl=sl_price,
            tp=tp_price
        )

    else:
        print("⏳ สัญญาณ: WAIT (ตลาดไม่ชัดเจน รอก่อน)")

    print("\n=================================================")


if __name__ == "__main__":
    print("🚀 เริ่มสตาร์ท ML Trading Bot...")
    main()

    schedule.every(10).seconds.do(main)  # ✅ แก้จาก seconds เป็น minutes

    print("\n⏳ บอทเข้าสู่โหมดเฝ้าตลาด (รันทุก 5 นาที) ... กด Ctrl+C เพื่อหยุด")

    while True:
        schedule.run_pending()
        time.sleep(1)