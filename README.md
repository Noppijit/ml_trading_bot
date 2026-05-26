# 🤖 ML Gold Trading Bot (XAUUSD)

โปรเจกต์บอทเทรดทองคำอัตโนมัติ ขับเคลื่อนด้วย Machine Learning (Scikit-Learn) สำหรับวิเคราะห์ทิศทางราคาและคำนวณจุดเข้าซื้อขาย ควบคู่กับระบบบริหารความเสี่ยง (Risk Management) ที่จำกัดความเสียหายไม่เกิน 2% ของพอร์ต

## 🌟 ฟีเจอร์หลัก (Features)
* **Data Ingestion:** ดึงข้อมูลราคาทองคำย้อนหลังอัตโนมัติผ่าน `yfinance`
* **Machine Learning Engine:** ใช้โมเดล AI ตัดสินใจทิศทางตลาด (ขึ้น/ลง)
* **Dynamic Risk Management:** คำนวณ Lot Size และตั้งจุด Stop Loss อัตโนมัติตามค่าความผันผวน (ATR)
* **Cross-Platform Ready:** โค้ดรองรับการทดสอบ (Dry Run) บน Mac และพร้อมเชื่อมต่อยิงคำสั่งจริงเข้า MetaTrader 5 (MT5) บน Windows VPS

## 📂 โครงสร้างโฟลเดอร์
```text
ML_TRADING_BOT/
├── models/                  # เก็บไฟล์โมเดล AI (.joblib)
├── notebooks/               # สมุดโน้ต Jupyter สำหรับเทรนและทดสอบข้อมูล
├── src/                     # โค้ดระบบหลัก (Core Engine)
│   ├── data_loader.py       # ระบบดึงข้อมูล
│   ├── features.py          # ระบบสร้าง Indicator
│   ├── risk_manager.py      # ระบบคำนวณความเสี่ยงและ Position Sizing
│   └── executor.py          # ระบบส่งคำสั่งเทรดเชื่อมต่อ MT5
├── .env                     # [สร้างเอง] ไฟล์เก็บ API Key และรหัสผ่าน
├── .gitignore               # กำหนดไฟล์ที่ไม่ต้องการอัปโหลดขึ้น Git
├── main.py                  # ไฟล์หลักสำหรับสั่งบอททำงาน
└── requirements.txt         # รายชื่อ Library ที่ต้องติดตั้ง

🚀 วิธีการติดตั้งและรันบอท (Setup)
Clone โปรเจกต์ และสร้าง Virtual Environment:
python -m venv .venv
source .venv/bin/activate  # สำหรับ Mac/Linux
# .venv\Scripts\activate   # สำหรับ Windows

ติดตั้ง Library:
pip install -r requirements.txt

รันการทำงาน (Daily Run):
python main.py

(หมายเหตุ: หากรันบน Mac ระบบจะเข้าสู่โหมด Dry Run เพื่อคำนวณจุดเข้าและ Stop Loss เท่านั้น จะไม่มีการส่งคำสั่งเข้า MT5)