import os
import sys
import joblib
import pandas as pd
from src.executor import execute_mt5_trade
import schedule
import time
from datetime import datetime, timedelta
import MetaTrader5 as mt5

# นำโฟลเดอร์ปัจจุบันเข้าระบบ เพื่อให้หาโฟลเดอร์ src เจอ
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import get_processed_data
from src.risk_manager import calculate_position_size

def main():
    print("=== 🚀 เริ่มต้นการทำงาน ML Trading Bot (Daily Run) ===")
    
    # -----------------------------------
    # 1. ตั้งค่าระบบสำหรับทองคำ (Config)
    # -----------------------------------
    SYMBOL = "XAUUSD"              # <--- เปลี่ยนเป็น GC=F
    ACCOUNT_BALANCE = 1000.00  
    RISK_PCT = 0.02
    
    # ดึงข้อมูลย้อนหลัง 1 ปีจนถึงวันนี้ (เพื่อให้มีจำนวนแท่งเทียนพอสำหรับคำนวณ Indicator)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # ---------------------------------------------------------
    # 2. เตรียมข้อมูล (Data Ingestion & Feature Engineering)
    # ---------------------------------------------------------
    print(f"\n[1/4] ดึงข้อมูล {SYMBOL} ล่าสุดจากตลาด...")
    try:
        df = get_processed_data(SYMBOL, interval='5m', num_bars=5000)
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการดึงข้อมูล: {e}")
        return

    # ---------------------------------------------------------
    # 3. โหลดสมองกล (AI Model)
    # ---------------------------------------------------------
    print("\n[2/4] โหลดโมเดล AI ที่ปรับจูนไว้...")
    model_path = os.path.join("models", "tuned_trading_model.joblib")
    
    if not os.path.exists(model_path):
        print(f"❌ ไม่พบไฟล์โมเดลที่: {model_path}")
        print("กรุณากลับไปรันโค้ดบันทึกโมเดลใน Jupyter Notebook ก่อนครับ")
        return
        
    model = joblib.load(model_path)
    
    # ให้ดึงชื่อและลำดับฟีเจอร์มาจากความจำของโมเดลโดยตรง
    features = model.feature_names_in_.tolist()
    
    # เช็คความพร้อมของข้อมูล
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
         print(f"❌ ข้อมูลขาดฟีเจอร์เหล่านี้: {missing_cols} (ตรวจสอบ data_loader.py)")
         return

    # ---------------------------------------------------------
    # 4. ทำนายทิศทางราคาล่าสุด (Prediction)
    # ---------------------------------------------------------
    print("\n[3/4] วิเคราะห์ข้อมูลแท่งเทียนล่าสุด เพื่อวางแผนเทรด...")
    
    latest_data = df.iloc[-1:] # ดึงข้อมูลแถวสุดท้าย (วันล่าสุด)
    latest_date = latest_data.index[0].strftime('%Y-%m-%d')
    
    X_latest = latest_data[features]
    prediction = model.predict(X_latest)[0]
    prob = model.predict_proba(X_latest)[0][1] # โอกาสที่ราคาจะ "ขึ้น" (คลาส 1)
    
    # --- 🌟 เพิ่มบรรทัดนี้ เพื่อให้ AI ฟันธงสัญญาณเทรด (1=BUY, 0=SELL) ---
    signal = model.predict(X_latest)[0]

    # --- ดึงค่า ATR และราคาปัจจุบัน (ย่อหน้าตรงกันแล้ว) ---
    current_atr = df['atr_5'].iloc[-1] 
    current_price = mt5.symbol_info_tick(SYMBOL).ask 

    print("\n[4/4] 🎯 แผนการเทรดวันนี้ (Action Plan)")

    # ===================================================
    # ส่วนการตัดสินใจและส่งคำสั่งเข้า MT5
    # ===================================================
    if signal == 1: 
        print("🟢 สัญญาณ: BUY (เข้าซื้อ)")
            # 1. คำนวณระยะ SL และ TP
        sl_dist = current_atr * 1.0  # SL = 1 เท่าของความผันผวน
        tp_dist = sl_dist * 1.5      # TP = 1.5 เท่าของความเสี่ยง (R:R 1:1.5)
            
            # 2. คำนวณราคาสำหรับขา BUY
        sl_price = current_price - sl_dist
        tp_price = current_price + tp_dist
            
        print(f"   - ราคาเข้า: {current_price}")
        print(f"   - เป้าหมาย (TP): {tp_price}")
        print(f"   - ตัดขาดทุน (SL): {sl_price}")
            
            # 3. ยิงคำสั่ง
        execute_mt5_trade(
                symbol=SYMBOL, 
                order_type=mt5.ORDER_TYPE_BUY, 
                volume=0.01,           
                price=current_price, 
                sl=sl_price, 
                tp=tp_price
            )

    elif signal == -1: 
        print("🔴 สัญญาณ: SELL (ขายชอร์ต)")
        current_price_bid = mt5.symbol_info_tick(SYMBOL).bid
            
            # 1. คำนวณระยะ SL และ TP
        sl_dist = current_atr * 1.0
        tp_dist = sl_dist * 1.5
            
            # 2. คำนวณราคาสำหรับขา SELL
        sl_price = current_price_bid + sl_dist
        tp_price = current_price_bid - tp_dist
            
        print(f"   - ราคาเข้า: {current_price_bid}")
        print(f"   - เป้าหมาย (TP): {tp_price}")
        print(f"   - ตัดขาดทุน (SL): {sl_price}")
            
            # 3. ยิงคำสั่ง
        execute_mt5_trade(
                symbol=SYMBOL, 
                order_type=mt5.ORDER_TYPE_SELL, 
                volume=0.01, 
                price=current_price_bid, 
                sl=sl_price, 
                tp=tp_price
            )
    else:
        print("⏳ สัญญาณ: WAIT (ตลาดไม่ชัดเจน รอก่อน)")

    print("\n=================================================")

# ... (โค้ดฟังก์ชัน def main(): ของคุณอยู่ด้านบน) ...
# ===================================================
        # 1. ฟังก์ชันเช็กลิมิตออเดอร์ (สูงสุด 8 ไม้)
        # ===================================================
def check_order_limit(symbol: str, max_orders: int = 8) -> bool:
            positions = mt5.positions_get(symbol=symbol)
            if positions is None:
                print(f"❌ ดึงข้อมูลออเดอร์ไม่สำเร็จ (Error: {mt5.last_error()})")
                return True # กันเหนียวไว้ก่อน
            
            current_orders = len(positions)
            print(f"📊 ปัจจุบันมีออเดอร์ {symbol} เปิดอยู่: {current_orders}/{max_orders} ไม้")
            
            if current_orders >= max_orders:
                return True # ครบ 8 ไม้แล้ว ห้ามเปิดเพิ่ม
            return False # ยังไม่ครบ เปิดเพิ่มได้

        # ===================================================
        # 2. เตรียมข้อมูลราคาและระยะ SL/TP (แบบ Point)
        # ===================================================
        # ดึงราคาปัจจุบัน
            current_price = mt5.symbol_info_tick(SYMBOL).ask 
            current_price_bid = mt5.symbol_info_tick(SYMBOL).bid
        
        # ดึงขนาด 1 Point ของโบรกเกอร์ (เพื่อกันปัญหาทศนิยมผิดพลาด)
            symbol_point = mt5.symbol_info(SYMBOL).point
        
        # ตั้งค่าระยะตามที่คุณต้องการ
            sl_points = 350
            tp_points = 1000
        
        # แปลง Point เป็นระยะราคาจริง
            sl_dist = sl_points * symbol_point
            tp_dist = tp_points * symbol_point

            print("\n[4/4] 🎯 แผนการเทรดวันนี้ (Action Plan)")

        # ===================================================
        # 3. 🛡️ สแกนเช็กออเดอร์ก่อนเทรด
        # ===================================================
            if check_order_limit(SYMBOL, max_orders=8):
                print(f"⚠️ ข้ามการทำงาน: ออเดอร์เต็มลิมิต 8 ไม้แล้ว รอปิดกำไร/ขาดทุนก่อนครับ")
            return # จบการทำงานรอบนี้ทันที

        # ===================================================
        # 4. ตัดสินใจและส่งคำสั่งเข้า MT5
        # ===================================================
            if signal == 1: 
            # ... (โค้ดคำนวณราคา BUY) ...
            
                execute_mt5_trade(
                symbol=SYMBOL, 
                order_type=mt5.ORDER_TYPE_BUY, 
                volume=0.01,           
                price=current_price, 
                sl=sl_price, 
                tp=tp_price
            )   # <--- 🔴 ต้องมีวงเล็บปิดตรงนี้ด้วยครับ

            elif signal == -1: 
            # ... (โค้ดคำนวณราคา SELL) ...
            
                execute_mt5_trade(
                symbol=SYMBOL, 
                order_type=mt5.ORDER_TYPE_SELL, 
                volume=0.01, 
                price=current_price_bid, 
                sl=sl_price, 
                tp=tp_price
            )   # <--- 🔴 และต้องมีวงเล็บปิดตรงนี้ก่อนถึง else ครับ

            else:
                print("⏳ สัญญาณ: WAIT (ตลาดไม่ชัดเจน รอก่อน)")
if __name__ == "__main__":
    print("🚀 เริ่มสตาร์ท ML Trading Bot...")
    
    # สั่งให้รันทันที 1 รอบตอนเพิ่งเปิดโปรแกรม
    main()
    
    # ==========================================
    # โหมด AUTO: ตั้งเวลาให้บอททำงานเอง
    # ==========================================
    # สมมติว่าคุณเทรดกราฟ 5 นาที (5m) ก็ให้บอทตื่นมาเช็กตลาดทุกๆ 5 นาทีครับ
    schedule.every(5).seconds.do(main)
    
    print("\n⏳ บอทเข้าสู่โหมดเฝ้าตลาด (รันทุก 5 วินาที) ... กด Ctrl+C เพื่อหยุด")
    
    # ลูปอมตะ (ให้โปรแกรมรันค้างไว้ไม่ปิดตัวเอง)
    while True:
        schedule.run_pending()
        time.sleep(1) # ให้คอมพิวเตอร์พักหายใจ 1 วินาที จะได้ไม่กิน CPU