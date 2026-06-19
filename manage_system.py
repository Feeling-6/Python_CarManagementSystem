import os
import sqlite3
from datetime import datetime
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# 导入开源项目的车牌识别模块
from plate_recognition.plate_rec import init_model, get_plate_result

# 数据库文件名称
DB_NAME = "parking_management.db"

# ================= 模型初始化 =================
# 自动选择 GPU 或 CPU 计算加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 YOLOv8 定位模型
YOLO_MODEL_PATH = "yolov8s.pt"
yolo_model = YOLO(YOLO_MODEL_PATH) if os.path.exists(YOLO_MODEL_PATH) else None

# 初始化专用车牌识别模型
REC_MODEL_PATH = "plate_rec_color.pth"
plate_rec_model = init_model(device, REC_MODEL_PATH, is_color=True) if os.path.exists(REC_MODEL_PATH) else None
# ==============================================


def init_db():
    """初始化 SQLite 数据库，创建记录表"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            fee REAL,
            status TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def recognize_plate(image_path):
    """使用 YOLOv8 定位车牌，并使用专用识别模型提取文字"""
    if not os.path.exists(image_path):
        print(f"【错误】文件 {image_path} 不存在。")
        return None
        
    if yolo_model is None or plate_rec_model is None:
        print("【错误】模型加载不完整，无法进行识别。")
        return None

    print(f"正在识别 {image_path} ...")
    
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("【错误】图片读取失败，请检查文件格式。")
        return None

    # 第一步：YOLO 检测车牌位置
    results = yolo_model(img, verbose=False)
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        print("【失败】YOLO 未在该图片中检测到车牌区域。")
        return None
        
    # 取置信度最高的一个预测框坐标
    box = boxes[0].xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box
    
    # 裁剪车牌区域（已移除外扩和放大逻辑，直接送入专用模型）
    plate_img = img[y1:y2, x1:x2]
    
    # 第二步：调用原项目接口进行字符识别
    try:
        # 原接口返回四个值：车牌号, 识别概率, 颜色, 颜色概率
        plate_number, rec_prob, plate_color, color_conf = get_plate_result(plate_img, device, plate_rec_model, is_color=True)
    except Exception as e:
        print(f"【失败】识别模块发生异常：{e}")
        return None
        
    if not plate_number:
        print("【失败】未能识别出任何字符。")
        return None
        
    # 已移除正则过滤逻辑，相信专用模型的原生输出
    return plate_number

def handle_enter(image_path):
    """处理车辆进入逻辑"""
    plate_number = recognize_plate(image_path)
    if not plate_number:
        return
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 检查该车辆是否已经在场内
    cursor.execute("SELECT id FROM records WHERE plate_number = ? AND status = 'IN'", (plate_number,))
    if cursor.fetchone():
        print(f"【提示】车牌号【{plate_number}】已在停车场内，无需重复记录。")
        conn.close()
        return
        
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO records (plate_number, entry_time, status) VALUES (?, ?, 'IN')",
        (plate_number, now_str)
    )
    conn.commit()
    conn.close()
    print(f"【成功】车牌号【{plate_number}】于 {now_str} 成功入场。")

def handle_exit(image_path):
    """处理车辆离开逻辑"""
    plate_number = recognize_plate(image_path)
    if not plate_number:
        return
        
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 查找场内的入场记录
    cursor.execute("SELECT id, entry_time FROM records WHERE plate_number = ? AND status = 'IN'", (plate_number,))
    record = cursor.fetchone()
    
    if not record:
        print(f"【提示】未找到车牌号【{plate_number}】的场内入场记录。")
        conn.close()
        return
        
    record_id, entry_time_str = record
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
    
    # 计算停留时间（分钟）
    duration = now - entry_time
    duration_minutes = duration.total_seconds() / 60
    
    # 计费规则：假设每小时 5 元
    fee = round((duration_minutes / 60) * 5.0, 2)
    
    # 更新记录
    cursor.execute(
        "UPDATE records SET exit_time = ?, fee = ?, status = 'OUT' WHERE id = ?",
        (now_str, fee, record_id)
    )
    conn.commit()
    conn.close()
    
    print(f"【成功】车牌号【{plate_number}】已记录出场。")
    print(f"  - 入场时间：{entry_time_str}")
    print(f"  - 出场时间：{now_str}")
    print(f"  - 停留时间：{int(duration_minutes)} 分钟")
    print(f"  - 应缴费用：{fee} 元")

def pad_string(text, width):
    """
    处理中英文混合排版的对齐问题。
    计算字符串的实际显示长度：中文字符算作 2 个宽度，其他算作 1 个。
    """
    display_length = sum(2 if ord(c) > 127 else 1 for c in str(text))
    return str(text) + ' ' * max(0, width - display_length)

def handle_list():
    """查看停车场内的车辆和停留时间"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT plate_number, entry_time FROM records WHERE status = 'IN'")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("当前停车场内无在场车辆。")
        return
        
    print("\n===================== 当前在场车辆列表 =====================")
    # 彻底抛弃 \t 制表符，统一使用 pad_string 精确补齐空格
    print(f"{pad_string('车牌号', 16)}{pad_string('入场时间', 26)}{'当前已停留时间'}")
    print("-" * 60)
    
    now = datetime.now()
    for plate_number, entry_time_str in rows:
        entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
        duration = now - entry_time
        duration_minutes = int(duration.total_seconds() / 60)
        
        hours = duration_minutes // 60
        minutes = duration_minutes % 60
        duration_str = f"{hours}小时{minutes}分钟" if hours > 0 else f"{minutes}分钟"
        
        # 保持和表头完全一致的宽度参数：16 和 26
        print(f"{pad_string(plate_number, 16)}{pad_string(entry_time_str, 26)}{duration_str}")
    print("============================================================\n")

def handle_clear():
    """清空场内所有车辆"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM records WHERE status = 'IN'")
    count = cursor.fetchone()[0]
    
    if count == 0:
        print("【提示】当前停车场内无在场车辆，无需清空。")
        conn.close()
        return
        
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute(
        "UPDATE records SET exit_time = ?, fee = 0.0, status = 'OUT' WHERE status = 'IN'",
        (now_str,)
    )
    conn.commit()
    conn.close()
    
    print(f"【成功】已强制清空场内所有车辆，共计移除 {count} 辆车。")

def print_help():
    """显示命令帮助菜单"""
    print("\n可用命令列表:")
    print("  Enter <图片路径>  - 记录车辆入场")
    print("  Exit <图片路径>   - 记录车辆出场")
    print("  List              - 查看场内车辆及当前停留时间")
    print("  Clear             - 清空场内所有车辆记录")
    print("  Quit              - 退出系统")

def main():
    init_db()
    print("=== 停车场命令行车辆管理系统 ===")
    print_help()
    
    while True:
        try:
            user_input = input("\n请输入命令 > ").strip()
            if not user_input:
                continue
                
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            
            if command == "quit":
                print("程序已退出。")
                break
            elif command == "list":
                handle_list()
            elif command == "clear":
                handle_clear()
            elif command in ["enter", "exit"]:
                if len(parts) < 2:
                    print(f"【错误】{parts[0]} 命令需要提供图片路径。示例：{parts[0]} 苏A12345.jpg")
                    continue
                image_path = parts[1]
                if command == "enter":
                    handle_enter(image_path)
                else:
                    handle_exit(image_path)
            else:
                print("【错误】未知命令。")
        except KeyboardInterrupt:
            print("\n程序已退出。")
            break

if __name__ == "__main__":
    main()