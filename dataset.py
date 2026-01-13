import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class PinDataset(Dataset):
    def __init__(self, data_list, processor, max_length=128):
        self.data_list = data_list  # 格式: [{"image_path": "...", "num": "1", "name": "CLK"}]
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        
        # 1. 图像预处理
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # 2. 构建目标序列: <s_pin_num>1</s_pin_num><s_pin_name>CLK</s_pin_name>
        target_sequence = (
            f"<s_pin_number>{item['num']}</s_pin_number>"
            f"<s_pin_name>{item['name']}</s_pin_name></s>"
        )
        
        # 3. 文本 Tokenize
        labels = self.processor.tokenizer(
            target_sequence, 
            add_special_tokens=False, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        ).input_ids.squeeze()

        # 将 pad_token 设为 -100，以便在计算 Loss 时忽略它
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {"pixel_values": pixel_values, "labels": labels}