# mm_pinnnet.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ----------------------------
# 字符映射（必须与训练一致）
# ----------------------------
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR_TO_IDX = {"<PAD>": 0}
for i, c in enumerate(CHARS):
    CHAR_TO_IDX[c] = i + 1
CHAR_TO_IDX["<EOS>"] = len(CHARS) + 1
VOCAB_SIZE = len(CHAR_TO_IDX)
IDX_TO_CHAR = {v: k for k, v in CHAR_TO_IDX.items()}
EOS_IDX = CHAR_TO_IDX["<EOS>"]
PAD_IDX = CHAR_TO_IDX["<PAD>"]

MAX_PINS = 16      # 最大引脚数
MAX_NAME_LEN = 6   # 名称最大长度（含<EOS>）

# ----------------------------
# 1. 数据集类：读取 dataset/images 和 dataset/annotations
# ----------------------------
class SymbolPinDataset(Dataset):
    def __init__(self, root_dir='dataset', img_size=224):
        self.root_dir = root_dir
        self.img_size = img_size
        self.image_dir = os.path.join(root_dir, 'images')
        self.ann_dir = os.path.join(root_dir, 'annotations')

        # 获取所有 JSON 文件
        self.ann_files = sorted([f for f in os.listdir(self.ann_dir) if f.endswith('.json')])
        assert len(self.ann_files) > 0, f"No JSON files found in {self.ann_dir}"

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.ann_files)

    def __getitem__(self, idx):
        ann_path = os.path.join(self.ann_dir, self.ann_files[idx])
        with open(ann_path, 'r') as f:
            ann = json.load(f)

        image_id = ann['image_id']
        img_path = os.path.join(self.image_dir, f"{image_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, f"{image_id}.jpg")

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # 按 pin_number 排序并填充到 MAX_PINS
        pin_dict = {}
        for pin in ann['pins']:
            pin_num = pin['pin_number']
            name = pin['name'].upper()
            pin_dict[pin_num] = name

        pin_ids = []
        pin_names = []
        for i in range(1, MAX_PINS + 1):
            if i in pin_dict:
                pin_ids.append(i)
                pin_names.append(pin_dict[i])
            else:
                pin_ids.append(0)
                pin_names.append("PAD")

        # 编码为 token ID 序列
        name_seq = []
        for name in pin_names:
            if name == "PAD":
                seq = [PAD_IDX] * MAX_NAME_LEN
            else:
                chars = list(name) + ["<EOS>"]
                chars = (chars + ["<PAD>"] * MAX_NAME_LEN)[:MAX_NAME_LEN]
                seq = [CHAR_TO_IDX.get(c, PAD_IDX) for c in chars]
            name_seq.extend(seq)

        pin_ids = torch.tensor(pin_ids, dtype=torch.long)      # [16]
        name_seq = torch.tensor(name_seq, dtype=torch.long)    # [96]

        return image, pin_ids, name_seq


# ----------------------------
# 2. MM-PinNet 模型
# ----------------------------
class MM_PinNet(nn.Module):
    def __init__(self, max_pins=MAX_PINS, max_name_len=MAX_NAME_LEN, vocab_size=VOCAB_SIZE, d_model=256, nhead=8, num_layers=2):
        super().__init__()
        self.max_pins = max_pins
        self.max_name_len = max_name_len

        self.dropout = nn.Dropout(0.3)
        # 使用预训练 ResNet18 作为特征提取器
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]  # 输出 [B, 512, H/32, W/32]
        self.cnn = nn.Sequential(*modules)
        self.feat_proj = nn.Linear(512, d_model)  # 投影每个空间位置
        # 位置嵌入
        self.pin_pos_embed = nn.Embedding(max_pins, d_model // 2)
        self.char_pos_embed = nn.Embedding(max_name_len, d_model // 2)
        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # 输出头
        self.exist_head = nn.Linear(d_model, 1)
        self.char_head = nn.Linear(d_model, vocab_size)

    def forward(self, images):
        B = images.shape[0]
        feats = self.cnn(images)
        H, W = feats.shape[2], feats.shape[3]
        # 展平空间维度: [B, 512, H*W] -> [B, H*W, 512]
        feats = feats.flatten(2).transpose(1, 2)  # [B, H*W, 512]
        memory = self.dropout(self.feat_proj(feats))  # [B, H*W, d_model]
        # 准备解码器查询
        device = images.device
        pin_ids = torch.arange(self.max_pins, device=device).unsqueeze(1).repeat(1, self.max_name_len).flatten()
        char_ids = torch.arange(self.max_name_len, device=device).repeat(self.max_pins)
        pin_embed = self.pin_pos_embed(pin_ids)
        char_embed = self.char_pos_embed(char_ids)
        query = torch.cat([pin_embed, char_embed], dim=-1).unsqueeze(0).repeat(B, 1, 1)
        # Transformer 解码
        decoded = self.decoder(query, memory)
        decoded = self.dropout(decoded)
        # 分支输出
        decoded_reshaped = decoded.view(B, self.max_pins, self.max_name_len, -1)
        pin_feat = decoded_reshaped.mean(dim=2)
        pin_exists_logits = self.exist_head(pin_feat).squeeze(-1)
        char_logits = self.char_head(decoded)

        return pin_exists_logits, char_logits


# ----------------------------
# 3. 训练函数
# ----------------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集路径
    train_dataset = SymbolPinDataset(root_dir='./dataset/train', img_size=224)
    val_dataset = SymbolPinDataset(root_dir='./dataset/val', img_size=224)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = MM_PinNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    char_criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    exist_criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0

    def evaluate(model, dataloader, device):
        model.eval()
        total_pins_exist = 0      # 真实存在的引脚数
        correct_exist = 0         # 存在性判断正确的数量
        correct_name_exact = 0    # 名称完全正确的数量（仅对真实存在的引脚）

        with torch.no_grad():
            for images, pin_ids, name_seqs in dataloader:
                images = images.to(device)
                pin_ids = pin_ids.to(device)      # [B, 16]
                name_seqs = name_seqs.to(device)  # [B, 96]

                pin_exists_logits, char_logits = model(images)
                pred_exists = (torch.sigmoid(pin_exists_logits) > 0.5)  # [B, 16]
                char_preds = torch.argmax(char_logits, dim=-1)          # [B, 96]

                B = images.size(0)
                for b in range(B):
                    for i in range(MAX_PINS):
                        true_pin_id = pin_ids[b, i].item()
                        pred_exist = pred_exists[b, i].item()

                        # 存在性准确率
                        true_exist = (true_pin_id != 0)
                        if pred_exist == true_exist:
                            correct_exist += 1

                        # 仅当真实存在时，才评估名称
                        if true_exist:
                            total_pins_exist += 1

                            # 解码真实名称
                            start = i * MAX_NAME_LEN
                            end = start + MAX_NAME_LEN
                            true_seq = name_seqs[b, start:end]
                            pred_seq = char_preds[b, start:end]

                            # 提取真实字符串（到<EOS>或<PAD>为止）
                            true_chars = []
                            for idx in true_seq:
                                idx = idx.item()
                                if idx == EOS_IDX or idx == PAD_IDX:
                                    break
                                true_chars.append(IDX_TO_CHAR.get(idx, ''))
                            true_name = ''.join(true_chars)

                            # 提取预测字符串
                            pred_chars = []
                            for idx in pred_seq:
                                idx = idx.item()
                                if idx == EOS_IDX or idx == PAD_IDX:
                                    break
                                pred_chars.append(IDX_TO_CHAR.get(idx, ''))
                            pred_name = ''.join(pred_chars)

                            # 精确匹配
                            if pred_name == true_name:
                                correct_name_exact += 1

        exist_acc = correct_exist / (len(dataloader.dataset) * MAX_PINS)
        name_exact_acc = correct_name_exact / total_pins_exist if total_pins_exist > 0 else 0.0
        return exist_acc, name_exact_acc

    for epoch in range(80):
        # ---------- 训练 ----------
        model.train()
        total_loss = 0
        for images, pin_ids, name_seqs in train_loader:
            images = images.to(device)
            pin_ids = pin_ids.to(device)
            name_seqs = name_seqs.to(device)

            optimizer.zero_grad()
            pin_exists_logits, char_logits = model(images)

            char_loss = char_criterion(char_logits.transpose(1, 2), name_seqs)
            pin_exists_target = (pin_ids != 0).float()
            exist_loss = exist_criterion(pin_exists_logits, pin_exists_target)

            loss = 0.8 * char_loss + 0.2 * exist_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ---------- 评估训练集准确率 ----------
        train_exist_acc, train_name_acc = evaluate(model, train_loader, device)
        # ---------- 评估验证集准确率 ----------
        val_exist_acc, val_name_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Exist Acc: {train_exist_acc:.2%} | "
              f"Train Name Acc: {train_name_acc:.2%} | "
              f"Val Exist Acc: {val_exist_acc:.2%} | "
              f"Val Name Acc: {val_name_acc:.2%}")

        # 保存最佳模型
        if val_name_acc > best_val_acc:
            best_val_acc = val_name_acc
            torch.save(model.state_dict(), 'mm_pinnnet_best.pth')

    # 也保存最终模型
    torch.save(model.state_dict(), 'mm_pinnnet_final.pth')
    print("Training and evaluation finished.")


# ----------------------------
# 4. 推理函数
# ----------------------------
def inference(image_path, model_path='mm_pinnnet_final.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MM_PinNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pin_exists_logits, char_logits = model(input_img)
        # from attention_visualize import visualize_all_attentions
        # visualize_all_attentions(image, cross_attn_weights[0])
        pin_exists = torch.sigmoid(pin_exists_logits).cpu().numpy()[0]
        preds = torch.argmax(char_logits, dim=-1).cpu().numpy().flatten()

    result = {}
    for i in range(MAX_PINS):
        if pin_exists[i] > 0.5:
            start = i * MAX_NAME_LEN
            decoded_chars = []
            for idx in preds[start:start+MAX_NAME_LEN]:
                if idx == PAD_IDX or idx == EOS_IDX:
                    break
                decoded_chars.append(IDX_TO_CHAR.get(idx, ''))
            name = ''.join(decoded_chars)
            if name.strip():
                result[str(i + 1)] = name

    return result

# ----------------------------
# 5. 主程序入口
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='infer', choices=['train', 'infer'])
    parser.add_argument('--image', type=str, default='./dataset/val/images/chip_0002.png')
    args = parser.parse_args()

    if args.mode == 'train':
        print("Training on dataset...")
        train()
    elif args.mode == 'infer':
        result = inference(args.image)
        print("Predicted pins:")
        print(json.dumps(result, indent=2))