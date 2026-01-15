import os
import json
import random
from PIL import Image, ImageDraw, ImageFont

# ==================== 配置 ====================
OUTPUT_DIR = "./dataset2/val"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "annotations"), exist_ok=True)

IMG_SIZE = 256
FONT_SIZE = 10
CHIP_WIDTH = 60
CHIP_HEIGHT = 180  # 足够容纳16引脚
CHIP_X = (IMG_SIZE - CHIP_WIDTH) // 2
CHIP_Y = (IMG_SIZE - CHIP_HEIGHT) // 2

# 你的引脚名称库
PIN_NAMES_POOL = [
    # ========== 电源与地 (Power & Ground) ==========
    "VCC", "VDD", "VSS", "GND", "VEE", "VPP", "VIN", "VOUT", "3V3", "5V",
    "1V8", "1V2", "2V5", "12V", "AVDD", "DVDD", "AGND", "DGND", "PGND", "SGND",
    "VBAT", "VUSB", "VREG", "VREF", "VFB", "VSENSE", "VSYS", "VANA", "VDIG", "VIO",
    "VDDIO", "VDDD", "VDDA", "VDDL", "VDDQ", "VTT", "VPP", "VNEG", "VPOS", "VBIAS",

    # ========== 数字控制信号 (Digital Control) ==========
    "EN", "OE", "CE", "CS", "RST", "RESET", "PWR", "ON", "OFF", "STBY",
    "SHDN", "SD", "PD", "WP", "HOLD", "TRIG", "ALERT", "INT", "IRQ", "FLAG",
    "READY", "BUSY", "DONE", "START", "STOP", "MODE", "SEL", "SELECT", "CONFIG", "PROG",
    "BOOT", "BOOT0", "BOOT1", "TEST", "TST", "MONITOR", "FAULT", "ERROR", "OK", "GOOD",

    # ========== 时钟与同步 (Clock & Sync) ==========
    "CLK", "SCK", "MCLK", "BCLK", "WCLK", "XTAL1", "XTAL2", "OSC", "REFCLK",
    "XIN", "XOUT", "CLKIN", "CLKOUT", "SYSCLK", "CORECLK", "PLLCLK", "FCLK", "HCLK", "PCLK",
    "SCLK", "I2S_CLK", "LRCLK", "MFP_CLK", "RTC_CLK", "CLK_32K", "CLK_27M", "REF_OSC", "OSC_IN", "OSC_OUT",

    # ========== 数据与总线 (Data & Bus) ==========
    "DATA", "DQ", "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7",
    "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "ADDR", "AD0", "AD1",
    "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15",
    "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15",
    "DB0", "DB1", "DB2", "DB3", "DB4", "DB5", "DB6", "DB7",
    "AB0", "AB1", "AB2", "AB3", "AB4", "AB5", "AB6", "AB7",
    "DATA0", "DATA1", "DATA2", "DATA3", "DATA4", "DATA5", "DATA6", "DATA7",

    # ========== 通用输入输出 (GPIO & I/O) ==========
    "IN", "OUT", "IO", "I0", "I1", "I2", "I3", "O0", "O1", "O2", "O3",
    "GPIO", "DI", "DO", "SI", "SO", "SDI", "SDO", "MISO", "MOSI", "SCL", "SDA",
    "GPIO0", "GPIO1", "GPIO2", "GPIO3", "GPIO4", "GPIO5", "GPIO6", "GPIO7",
    "GPIO8", "GPIO9", "GPIO10", "GPIO11", "GPIO12", "GPIO13", "GPIO14", "GPIO15",
    "IO0", "IO1", "IO2", "IO3", "IO4", "IO5", "IO6", "IO7",
    "PORTA", "PORTB", "PORTC", "PORTD", "PIN_A", "PIN_B", "PIN_C", "PIN_D",

    # ========== 模拟信号 (Analog Signals) ==========
    "AIN", "AOUT", "VINP", "VINN", "VREF", "FB", "COMP", "SENSE", "ISNS",
    "AN0", "AN1", "AN2", "AN3", "AN4", "AN5", "AN6", "AN7",
    "AIN0", "AIN1", "AIN2", "AIN3", "AIN4", "AIN5", "AIN6", "AIN7",
    "AUX", "TEMP", "THERM", "VTEMP", "VMEAS", "VMON", "IMON", "DIFFP", "DIFFN",

    # ========== 通信接口 - UART / Serial ==========
    "TX", "RX", "TXD", "RXD", "CTS", "RTS", "DSR", "DTR", "DCD", "RI",
    "UART_TX", "UART_RX", "SERIAL_TX", "SERIAL_RX", "SCI_TX", "SCI_RX",

    # ========== 通信接口 - CAN / LIN / Automotive ==========
    "CANH", "CANL", "RS485", "LIN", "SWDIO", "SWCLK", "TMS", "TDI", "TDO", "TCK",
    "CAN_TX", "CAN_RX", "CAN_H", "CAN_L", "LIN_IN", "LIN_OUT", "SENT", "PWM", "WAKE",

    # ========== 通信接口 - I2C / SPI ==========
    "SCL", "SDA", "SCLK", "MOSI", "MISO", "SS", "NSS", "CS0", "CS1", "SPI_CS",
    "I2C_SCL", "I2C_SDA", "SPI_CLK", "SPI_MOSI", "SPI_MISO", "SPI_SS",

    # ========== 通信接口 - USB ==========
    "USB_DP", "USB_DM", "USB_VBUS", "USB_ID", "USB_OTG", "D+", "D-", "VBUS", "ID",

    # ========== 通信接口 - Ethernet / PHY ==========
    "ETH_TX+", "ETH_TX-", "ETH_RX+", "ETH_RX-", "MDIO", "MDC", "CRS", "COL", "RXER", "TXEN",
    "RMII_TXD0", "RMII_TXD1", "RMII_RXD0", "RMII_RXD1", "RMII_REF_CLK", "RMII_CRS_DV", "RMII_RX_ER", "RMII_TX_EN",

    # ========== 通信接口 - PCIe / High-Speed ==========
    "PCIE_CLK", "PCIE_RST", "PCIE_RX", "PCIE_TX", "REFCLK_P", "REFCLK_N", "PERST#", "WAKE#",

    # ========== 存储器接口 (Memory) ==========
    "WE", "OE", "UB", "LB", "CAS", "RAS", "CS0", "CS1", "CS2", "CS3",
    "BYTE", "WAIT", "RDY", "HOLD", "WP", "A16", "A17", "A18", "A19", "A20",
    "D16", "D17", "D18", "D19", "D20", "D21", "D22", "D23", "D24", "D25", "D26", "D27", "D28", "D29", "D30", "D31",

    # ========== 射频与无线 (RF & Wireless) ==========
    "ANT", "RF", "BIAS", "TEMP", "RSSI", "PA_EN", "LNA_EN", "TX_SW", "RX_SW", "FEM_CTRL",
    "GPS_ANT", "BT_ANT", "WIFI_ANT", "RF_IN", "RF_OUT", "LO", "IF", "MIXER", "VCO", "PLL",

    # ========== 电源管理 (Power Management) ==========
    "PG", "POK", "PGOOD", "UVLO", "OVLO", "THERM_SHDN", "ILIM", "SS", "TRACK", "SYNC",
    "FB", "COMP", "ISENSE", "VSENSE", "ADJ", "DIM", "PWM_DIM", "ENABLE", "SHUTDOWN", "RUN",

    # ========== 特殊功能与保留引脚 ==========
    "NC", "NO", "COM", "ANT", "RF", "BIAS", "TEMP", "TEST", "PROG", "BOOT",
    "RESERVED", "RSV", "UNUSED", "DNU", "EPAD", "EXPOSED_PAD", "PAD", "METAL", "SUBSTRATE", "BODY",

    # ========== FPGA / CPLD 专用 ==========
    "CLK0", "CLK1", "CLK2", "CLK3", "USER_CLK", "CONFIG_DONE", "INIT_DONE", "CRC_ERROR", "nSTATUS", "CONF_DONE",
    "MSEL0", "MSEL1", "MSEL2", "MSEL3", "nCONFIG", "nCE", "nCEO", "DATA0", "DCLK", "nSTATUS",

    # ========== 微控制器 (MCU) 专用 ==========
    "NRST", "nRST", "RST#", "POR", "BOD", "WDT", "SWO", "SWD", "JTAG", "TRACECLK",
    "TCK", "TMS", "TDI", "TDO", "TRST", "DBG", "ETM", "ITM", "SWO", "SWV",

    # ========== 音频接口 ==========
    "MIC_IN", "LINE_IN", "HP_OUT", "SPK_OUT", "AUD_IN", "AUD_OUT", "BCLK", "WS", "SD", "MCLK",
    "I2S_SD", "I2S_WS", "I2S_SCK", "PCM_CLK", "PCM_FS", "PCM_DIN", "PCM_DOUT",

    # ========== 视频接口 ==========
    "HSYNC", "VSYNC", "PCLK", "DE", "LCD_R0", "LCD_G0", "LCD_B0", "CAM_D0", "CAM_PCLK", "CAM_HSYNC",
    "MIPI_D0P", "MIPI_D0N", "MIPI_CLKP", "MIPI_CLKN", "LVDS_P", "LVDS_N",

    # ========== 传感器接口 ==========
    "SCL1", "SDA1", "SCL2", "SDA2", "INT1", "INT2", "DRDY", "CONVST", "EOC", "FSYNC",
    "ACCEL_X", "ACCEL_Y", "ACCEL_Z", "GYRO_X", "GYRO_Y", "GYRO_Z", "MAG_X", "MAG_Y", "MAG_Z"
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
            # 编号紧贴芯片左侧
            num_x = x_start - 5  # 留一点空隙
            # 名称放在编号左侧更远处
            name_x = num_x - 80
        else:
            x_start = CHIP_X + CHIP_WIDTH
            # 编号紧贴芯片右侧
            num_x = x_start + 5
            # 名称放在编号右侧更远处
            name_x = num_x + 20
        
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
    num_samples = 200  
    for i in range(num_samples):
        generate_chip_sample(i)
    print(f"\n✅ Done! Generated {num_samples} diverse DIP samples.")