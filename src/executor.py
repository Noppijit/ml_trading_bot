import sys

# ป้องกัน Error บน Mac: เราจะโหลด Library MT5 ก็ต่อเมื่อรันบน Windows เท่านั้น
if sys.platform == 'win32':
    import MetaTrader5 as mt5

def execute_mt5_trade(symbol: str, order_type: int, volume: float, price: float, sl: float, tp: float):
    """
    ฟังก์ชันส่งคำสั่งเข้า MT5 พร้อมตั้งค่า SL และ TP
    """
    # 1. เช็คข้อมูลของคู่เงินเพื่อดูว่าใช้ทศนิยมกี่ตำแหน่ง
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ ไม่พบข้อมูลคู่เงิน {symbol} ใน MT5")
        return None

    digits = symbol_info.digits  # ดึงจำนวนทศนิยม เช่น ทองคำมักจะเป็น 2 หรือ 3 ตำแหน่ง

    # 2. ปัดเศษทศนิยม (สำคัญมาก! ถ้าไม่ปัด MT5 จะปฏิเสธคำสั่ง)
    sl_rounded = round(sl, digits)
    tp_rounded = round(tp, digits)
    
    print(f"เตรียมส่งคำสั่ง {symbol} | SL: {sl_rounded} | TP: {tp_rounded}")

    # 3. สร้างชุดคำสั่ง (Request)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "price": price,
        "sl": sl_rounded,      # ใส่ค่า Stop Loss ที่ปัดเศษแล้ว
        "tp": tp_rounded,      # ใส่ค่า Take Profit ที่ปัดเศษแล้ว
        "deviation": 20,       # ยอมรับราคาที่คลาดเคลื่อนได้ 20 จุด
        "magic": 123456,       # เลขประจำตัวบอท
        "comment": "ML Bot with SL/TP",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # 4. ส่งคำสั่งเข้าเซิร์ฟเวอร์
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ ส่งคำสั่งไม่สำเร็จ! Error Code: {result.retcode}")
        # ถ้าติด Error 10016 แปลว่าระยะ SL/TP ใกล้ราคาปัจจุบันเกินไป
    else:
        print(f"✅ เปิดออเดอร์สำเร็จ! Ticket: {result.order}")
        
    return result