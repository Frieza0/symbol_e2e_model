import os
import json
import random
from PIL import Image, ImageDraw, ImageFont

# ==================== 配置 ====================
OUTPUT_DIR = "./dataset1/train"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "annotations"), exist_ok=True)

IMG_SIZE = 256
FONT_SIZE = 14
CHIP_WIDTH = 80
CHIP_HEIGHT = 180  # 足够容纳16引脚
CHIP_X = (IMG_SIZE - CHIP_WIDTH) // 2
CHIP_Y = (IMG_SIZE - CHIP_HEIGHT) // 2

# 你的引脚名称库
PIN_NAMES_POOL = [
    # 电源与地
    "VCC", "VDD", "VSS", "GND", "VEE", "VPP", "VIN", "VOUT", "3V3", "5V",
    
    # 数字控制信号
    "EN", "OE", "CE", "CS", "RST", "RESET", "PWR", "ON", "OFF", "STBY",
    "SHDN", "SD", "PD", "WP", "HOLD", "TRIG", "ALERT", "INT", "IRQ", "FLAG",
    
    # 时钟与同步
    "CLK", "SCK", "MCLK", "BCLK", "WCLK", "XTAL1", "XTAL2", "OSC", "REFCLK",
    
    # 数据与总线
    "DATA", "DQ", "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7",
    "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "ADDR", "AD0", "AD1",
    
    # 通用输入输出
    "IN", "OUT", "IO", "I0", "I1", "I2", "I3", "O0", "O1", "O2", "O3",
    "GPIO", "DI", "DO", "SI", "SO", "SDI", "SDO", "MISO", "MOSI", "SCL", "SDA",
    
    # 模拟信号
    "AIN", "AOUT", "VINP", "VINN", "VREF", "FB", "COMP", "SENSE", "ISNS",
    
    # 通信接口专用
    "TX", "RX", "TXD", "RXD", "CTS", "RTS", "DSR", "DTR", "DCD", "RI",
    "CANH", "CANL", "RS485", "LIN", "SWDIO", "SWCLK", "TMS", "TDI", "TDO", "TCK",
    
    # 特殊功能
    "BOOT", "PROG", "TEST", "NC", "NO", "COM", "ANT", "RF", "BIAS", "TEMP"
]

POSSIBLE_PIN_COUNTS = [6, 8, 10, 12, 14, 16]

def get_font():
    try:
        return ImageFont.truetype("arial.ttf", FONT_SIZE)
    except:
        return ImageFont.load_default()

def generate_chip_sample(idx):
    total_pins = random.choice(POSSIBLE_PIN_COUNTS)
    half = total_pins // 2
    
    if total_pins > len(PIN_NAMES_POOL):
        raise ValueError("Not enough pin names in pool!")
    selected_names = random.sample(PIN_NAMES_POOL, total_pins)
    
    pins = []
    for i in range(half):
        pins.append((i, i + 1, selected_names[i], True))
    for i in range(half):
        pin_id = half + i
        pin_num = total_pins - i
        pins.append((pin_id, pin_num, selected_names[pin_id], False))
    
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
    draw = ImageDraw.Draw(img)
    font = get_font()
    
    chip_rect = [CHIP_X, CHIP_Y, CHIP_X + CHIP_WIDTH, CHIP_Y + CHIP_HEIGHT]
    draw.rectangle(chip_rect, outline="black", width=2)
    arc_bbox = [CHIP_X + 10, CHIP_Y - 10, CHIP_X + CHIP_WIDTH - 10, CHIP_Y + 10]
    draw.arc(arc_bbox, start=0, end=180, fill="black", width=2)
    
    pin_step = CHIP_HEIGHT / (half - 1) if half > 1 else CHIP_HEIGHT
    ann_pins = []
    
    for pin_id, pin_num, name, is_left in pins:
        local_idx = pin_id if is_left else pin_id - half
        y = CHIP_Y + int(local_idx * pin_step) if half > 1 else CHIP_Y + CHIP_HEIGHT // 2
        
        if is_left:
            x_start = CHIP_X
            x_end = x_start - 25
            num_x = x_start - 10
            name_x = x_end - 30
        else:
            x_start = CHIP_X + CHIP_WIDTH
            x_end = x_start + 25
            num_x = x_start + 5
            name_x = x_end + 10
        
        # 计算编号与名称文本的大小
        num_size = draw.textsize(str(pin_num), font=font)
        name_size = draw.textsize(name, font=font)
        
        # 编号 bbox: 左上角(x, y-num_size[1]/2), 右下角(x+num_size[0], y+num_size[1]/2)
        bbox_pin = [num_x, y - num_size[1]//2, num_x + num_size[0], y + num_size[1]//2]
        # 名称 bbox: 类似地计算
        bbox_name = [name_x, y - name_size[1]//2, name_x + name_size[0], y + name_size[1]//2]
        
        draw.text((num_x, y - num_size[1]//2), str(pin_num), fill="black", font=font)
        draw.text((name_x, y - name_size[1]//2), name, fill="black", font=font)
        
        ann_pins.append({
            "pin_number": pin_num,
            "name": name,
            "bbox_pin": bbox_pin,
            "bbox_name": bbox_name
        })
    
    img_path = os.path.join(OUTPUT_DIR, "images", f"chip_{idx:04d}.png")
    img.save(img_path)
    
    ann = {
        "image_id": f"chip_{idx:04d}",
        "width": IMG_SIZE,
        "height": IMG_SIZE,
        "total_pins": total_pins,
        "pins": ann_pins
    }
    ann_path = os.path.join(OUTPUT_DIR, "annotations", f"chip_{idx:04d}.json")
    with open(ann_path, 'w') as f:
        json.dump(ann, f, indent=2)
    
    print(f"Generated chip_{idx:04d} with {total_pins} pins and bounding boxes for each pin number and name.")

if __name__ == "__main__":
    num_samples = 500  
    for i in range(num_samples):
        generate_chip_sample(i)
    print(f"\n✅ Done! Generated {num_samples} diverse DIP samples.")