import json
from pathlib import Path
from PIL import Image

# ====== 配置区 ======
raw_data_dir = Path("symbol_dataset/1")          # 原始数据目录（含 .json + .png/.jpg）
output_images_dir = Path("symbol_dataset/images")       # 输出图像目录
output_ann_dir = Path("symbol_dataset/annotations")     # 输出标注目录

# 支持的图像扩展名
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
# ===================

output_images_dir.mkdir(exist_ok=True)
output_ann_dir.mkdir(exist_ok=True)

# 获取所有 JSON 文件
json_files = [f for f in raw_data_dir.iterdir() if f.suffix.lower() == ".json"]

for json_path in sorted(json_files):
    # 推断图像文件路径（同名不同后缀）
    img_path = None
    for ext in IMG_EXTENSIONS:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            img_path = candidate
            break

    if not img_path or not img_path.exists():
        print(f"⚠️  跳过 {json_path.name}：未找到对应图像文件")
        continue

    # 读取图像尺寸
    try:
        with Image.open(img_path) as img:
            width, height = img.size
    except Exception as e:
        print(f"❌ 无法打开图像 {img_path}: {e}")
        continue

    # 读取 JSON 内容
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 无法解析 JSON {json_path}: {e}")
        continue

    # 提取 content（兼容有无外层）
    content = data.get("content", data)

    # 映射 package_type
    type_map = {
        "1side": "1side",
        "2side": "2side",
        "4side": "4side",
        "circle": "circle"
    }
    package_type = type_map.get(content.get("type", "2side"), "2side")

    # 转换 pins
    pins = []
    for pin in content.get("pins", []):
        pins.append({
            "pin_number": str(pin["number"]).strip(),
            "name": str(pin.get("name", "")).strip(),
            "bbox_pin": None,
            "bbox_name": None
        })

    # 使用图像文件名（不含扩展名）作为 image_id 和 file_name
    stem = img_path.stem
    file_name = img_path.name  # 保留原始扩展名，如 .png

    new_data = {
        "image_id": stem,
        "width": width,
        "height": height,
        "total_pins": len(pins),
        "package_type": package_type,
        "pins": pins,
        "file_name": file_name
    }

    # 保存新 JSON 到 annotations/
    ann_path = output_ann_dir / f"{stem}.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    # 复制图像到 images/（避免修改原始数据）
    dst_img = output_images_dir / file_name
    if dst_img != img_path:  # 避免覆盖自身
        dst_img.write_bytes(img_path.read_bytes())

    print(f"✅ {json_path.name} → {file_name} | {width}x{height} | {len(pins)} pins | {package_type}")

print("\n🎉 数据集转换完成！")