# mm_pinnnet.py
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import math
from collections import defaultdict
import matplotlib.pyplot as plt
# ----------------------------
# 1. 模拟数据集生成（SymbolPinDataset）
# ----------------------------
class SymbolPinDataset(Dataset):
    def __init__(self, root_dir, split='train', num_samples=1000, img_size=256):
        self.root_dir = root_dir
        self.split = split
        self.num_samples = num_samples
        self.img_size = img_size
        self.pin_names = ["VCC", "GND", "CLK", "DATA", "RESET", "EN", "OUT", "IN"]
        os.makedirs(root_dir, exist_ok=True)
        if split == 'train':
            self._generate_data()

    def _generate_data(self):
        for i in range(self.num_samples):
            img_path = os.path.join(self.root_dir, f"{i}.png")
            json_path = os.path.join(self.root_dir, f"{i}.json")
            if os.path.exists(img_path) and os.path.exists(json_path):
                continue

            # 创建空白图像
            img = Image.new('RGB', (self.img_size, self.img_size), 'white')
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()

            pin_count = random.choice([4, 6, 8])
            center = self.img_size // 2
            radius = self.img_size * 0.35
            pins = {}
            angle_step = 360 / pin_count

            for idx in range(pin_count):
                angle_deg = idx * angle_step + random.uniform(-5, 5)
                angle_rad = math.radians(angle_deg)
                x = center + radius * math.cos(angle_rad)
                y = center + radius * math.sin(angle_rad)

                pin_id = str(idx + 1)
                pin_name = random.choice(self.pin_names)
                pins[pin_id] = pin_name

                # 绘制引脚编号（靠近圆周）
                draw.text((x - 8, y - 25), pin_id, fill='black', font=font)
                # 绘制引脚名称（稍远离）
                draw.text((x - 12, y + 5), pin_name, fill='blue', font=font)

            # 保存
            img.save(img_path)
            with open(json_path, 'w') as f:
                json.dump(pins, f)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, f"{idx}.png")
        json_path = os.path.join(self.root_dir, f"{idx}.json")

        image = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        image = transform(image)

        with open(json_path, 'r') as f:
            pins = json.load(f)

        # 构建目标：固定最大8个引脚
        MAX_PINS = 8
        pin_ids = []
        pin_names = []
        for i in range(1, MAX_PINS + 1):
            pid = str(i)
            if pid in pins:
                pin_ids.append(i)
                pin_names.append(pins[pid])
            else:
                pin_ids.append(0)  # padding
                pin_names.append("PAD")  # special pad token for unused pins

        # --- 关键修改：字符映射包含 <EOS> ---
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
        char_to_idx["<PAD>"] = 0      # padding (unused positions)
        char_to_idx["<EOS>"] = len(chars) + 1  # e.g., 37

        name_seq = []
        for name in pin_names:
            if name == "PAD":
                # 整个引脚未使用：6 个 <PAD>
                seq = [char_to_idx["<PAD>"]] * 6
            else:
                # 真实名称：添加 <EOS>，然后填充到 6
                # 例如 "DATA" → ['D','A','T','A','<EOS>','<PAD>']
                seq_chars = list(name) + ["<EOS>"]
                # 截断或填充到最多 6 个字符（含 <EOS>）
                seq_chars = (seq_chars + ["<PAD>"] * 6)[:6]
                seq = [char_to_idx.get(c, 0) for c in seq_chars]  # 未知字符→<PAD>
            name_seq.extend(seq)

        name_seq = torch.tensor(name_seq, dtype=torch.long)  # [48]
        pin_ids = torch.tensor(pin_ids, dtype=torch.long)    # [8]

        return image, pin_ids, name_seq


# ----------------------------
# 2. MM-PinNet 模型定义
# ----------------------------
class MM_PinNet(nn.Module):
    def __init__(self, max_pins=8, max_name_len=6, vocab_size=38):
        super().__init__()
        self.max_pins = max_pins
        self.max_name_len = max_name_len
        self.vocab_size = vocab_size

        # 视觉编码器
        from torchvision.models import resnet18
        backbone = resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])

        # === 新增：引脚存在性预测头 ===
        self.pin_exist_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, max_pins)  # sigmoid 输出每个引脚是否存在
        )

        # 文本解码部分（不变）
        self.feat_proj = nn.Linear(512, 256)
        self.query_embed = nn.Embedding(max_pins * max_name_len, 256)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        self.char_classifier = nn.Linear(256, vocab_size)

    def forward(self, x):
        B = x.shape[0]
        feats = self.feature_extractor(x)  # [B, 512, 7, 7]

        # === 新增：预测每个引脚是否存在 ===
        pin_exists_logits = self.pin_exist_head(feats)  # [B, 8]

        # 文本解码（不变）
        pooled = torch.mean(feats, dim=[2, 3])  # [B, 512]
        total_queries = self.max_pins * self.max_name_len
        query = self.query_embed.weight[:total_queries].unsqueeze(0).repeat(B, 1, 1)
        memory = self.feat_proj(pooled).unsqueeze(1).repeat(1, total_queries, 1)
        decoded = self.decoder(query, memory)
        char_logits = self.char_classifier(decoded)  # [B, 48, V]

        return pin_exists_logits, char_logits  # ← 返回存在性 logits


# ----------------------------
# 3. 训练函数
# ----------------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SymbolPinDataset('./data/train', split='train', num_samples=500)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    model = MM_PinNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    char_criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <PAD>
    exist_criterion = nn.BCEWithLogitsLoss()  # binary classification

    model.train()
    for epoch in range(200):
        total_loss = 0
        for images, pin_ids, name_seqs in dataloader:
            images = images.to(device)
            pin_ids = pin_ids.to(device)  # [B, 8], 0 表示不存在
            name_seqs = name_seqs.to(device)

            optimizer.zero_grad()
            pin_exists_logits, char_logits = model(images)

            # Loss 1: 字符预测
            char_loss = char_criterion(char_logits.transpose(1, 2), name_seqs)

            # Loss 2: 引脚存在性（0→不存在，非0→存在）
            pin_exists_target = (pin_ids != 0).float()  # [B, 8]
            exist_loss = exist_criterion(pin_exists_logits, pin_exists_target)

            loss = char_loss + 0.5 * exist_loss  # 可调权重
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), 'mm_pinnnet.pth')


# ----------------------------
# 4. 推理函数
# ----------------------------
def inference(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MM_PinNet().to(device)
    model.load_state_dict(torch.load('mm_pinnnet.pth', map_location=device))
    model.eval()

    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pin_exists_logits, char_logits = model(input_img)
        pin_exists = torch.sigmoid(pin_exists_logits).cpu().numpy()[0]  # [8]
        preds = torch.argmax(char_logits, dim=-1).cpu().numpy().flatten()  # [48]

    # 字符映射（含<EOS>）
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    idx_to_char = {i + 1: c for i, c in enumerate(chars)}
    idx_to_char[0] = ''
    eos_idx = len(chars) + 1

    result = {}
    for i in range(8):
        # 方案：只有当模型认为引脚存在（>0.5）才解码
        if pin_exists[i] > 0.5:
            start = i * 6
            decoded_chars = []
            for idx in preds[start:start+6]:
                if idx == 0 or idx == eos_idx:
                    break
                decoded_chars.append(idx_to_char.get(idx, ''))
            name = ''.join(decoded_chars)
            if name:  # 非空
                result[str(i + 1)] = name

    return result


# ----------------------------
# 5. 主程序
# ----------------------------
if __name__ == "__main__":

    mode = "infer"
    if mode == "train":
        print("Starting training...")
        train()
    elif mode == "infer":
        img_path = 'data/train/0.png'
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
        result = inference(img_path)
        print("Extracted pins:", json.dumps(result, indent=2))
    else:
        print("Unknown mode. Use 'train' or 'infer'")
        