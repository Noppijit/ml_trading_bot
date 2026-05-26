import pandas as pd
import numpy as np

def calculate_position_size(account_balance: float, risk_pct: float, stop_loss_dist: float) -> float:
    """
    คำนวณจำนวนเหรียญ/หุ้นที่ควรซื้อ ตามความเสี่ยงที่รับได้ (เช่น ยอมเสียได้ 2% ของพอร์ต)
    """
    if stop_loss_dist <= 0:
        return 0
    risk_amount = account_balance * risk_pct
    return risk_amount / stop_loss_dist

def simulate_trading(model, X_test: pd.DataFrame, df_original: pd.DataFrame, 
                     initial_balance: float = 10000, risk_per_trade: float = 0.02, 
                     atr_multiplier: float = 1.0,  # 📉 เปลี่ยนค่าเริ่มต้นจาก 1.5 เป็น 1.0 ให้ SL สั้นลง
                     rr_ratio: float = 1.5,        # 🎯 เพิ่ม Risk:Reward Ratio (กำไร 1.5 เท่าของขาดทุน)
                     prob_threshold: float = 0.55):
    """
    จำลองการเทรดโดยใช้ผลทำนายจากโมเดล ผนวกกับระบบ Risk Management แบบ ATR
    """
    print(f"Starting balance: ${initial_balance:,.2f}")
    
    probs = model.predict_proba(X_test)[:, 1]
    trades = []
    current_balance = initial_balance
    test_dates = X_test.index

    for i, current_date in enumerate(test_dates):
        prob = probs[i]
        
        if prob >= prob_threshold:
            try:
                entry_price = df_original.loc[current_date, 'close']
                
                # --- ⚠️ การดึงค่า ATR (ต้องเช็คชื่อคอลัมน์ให้ตรงกับ data_loader ของคุณ) ---
                # ถ้าตอน Train คุณใช้ atr_5 ให้เปลี่ยนบรรทัดล่างเป็น 'atr_5' นะครับ
                atr = df_original.loc[current_date, 'atr_14'] 
                
                # 1. คำนวณ Stop Loss (SL) ให้สั้นลงตาม atr_multiplier ที่เราปรับ
                sl_dist = atr * atr_multiplier
                sl_price = entry_price - sl_dist
                
                # 2. คำนวณ Take Profit (TP) ตามสัดส่วน RR Ratio
                tp_dist = sl_dist * rr_ratio
                tp_price = entry_price + tp_dist
                
                # 3. คำนวณ Position Size
                pos_size = calculate_position_size(current_balance, risk_per_trade, sl_dist)
                investment_value = pos_size * entry_price
                
                if investment_value > current_balance:
                    pos_size = current_balance / entry_price
                
                # --- การจำลองผลลัพธ์ในแท่งถัดไป ---
                next_date_idx = df_original.index.get_loc(current_date) + 1
                next_date = df_original.index[next_date_idx]
                
                next_high = df_original.loc[next_date, 'high'] # ดึงจุดสูงสุดแท่งถัดไปมาเช็ค TP
                next_low = df_original.loc[next_date, 'low']
                next_close = df_original.loc[next_date, 'close']
                
                # 🎯 ตรวจสอบว่าชน SL หรือ TP ก่อนกัน (แบบจำลองอย่างง่าย)
                if next_low <= sl_price:
                    exit_price = sl_price
                    reason = "Stop Loss"
                elif next_high >= tp_price:
                    exit_price = tp_price
                    reason = "Take Profit"
                else:
                    exit_price = next_close
                    reason = "Time Exit (Close)"
                
                pnl = (exit_price - entry_price) * pos_size
                current_balance += pnl
                
                trades.append({
                    'entry_date': current_date,
                    'exit_date': next_date,
                    'action': 'BUY',
                    'entry_price': round(entry_price, 2),
                    'sl_price': round(sl_price, 2),  # บันทึก SL ไว้ดู
                    'tp_price': round(tp_price, 2),  # บันทึก TP ไว้ดู
                    'exit_price': round(exit_price, 2),
                    'position_size': round(pos_size, 4),
                    'pnl': round(pnl, 2),
                    'balance': round(current_balance, 2),
                    'exit_reason': reason,
                    'model_prob': round(prob, 2)
                })
                
            except IndexError:
                pass

    trade_log = pd.DataFrame(trades)
    
    if not trade_log.empty:
        total_trades = len(trade_log)
        win_trades = len(trade_log[trade_log['pnl'] > 0])
        win_rate = (win_trades / total_trades) * 100
        net_profit = current_balance - initial_balance
        
        print("\n--- Backtest Results ---")
        print(f"Total Trades : {total_trades}")
        print(f"Win Rate     : {win_rate:.2f}%")
        print(f"Net Profit   : ${net_profit:,.2f}")
        print(f"Final Balance: ${current_balance:,.2f}")
        
        # เพิ่มการแสดงสถิติว่าจบออเดอร์ด้วยเหตุผลอะไรบ้าง
        print("\n--- Exit Reasons ---")
        print(trade_log['exit_reason'].value_counts().to_string())
    else:
        print("\nNo trades executed.")
        
    return trade_log