import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SiameseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.InstanceNorm2d(64))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1, stride=2), nn.ReLU())

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return x3

class StackingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SiameseEncoder()

        self.up1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)

        self.score_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, stack):
        b, n, c, h, w = stack.shape
        x = stack.view(b * n, c, h, w)
        feat_individual = self.encoder.layer1(x)
        feat_final = self.encoder(x)

        _, cf, hf, wf = feat_final.shape
        fused, _ = torch.max(feat_final.view(b, n, cf, hf, wf), dim=1)

        d1 = torch.relu(self.up1(fused))
        d2 = torch.relu(self.up2(d1))

        if d2.shape[2:] != (h, w):
            d2 = F.interpolate(d2, size=(h, w), mode="bilinear", align_corners=False)

        context_expanded = (
            d2.unsqueeze(1).expand(-1, n, -1, -1, -1).reshape(b * n, 64, h, w)
        )

        logits = self.score_head(context_expanded)

        logits = logits.view(b, n, h, w)
        weights = torch.softmax(logits, dim=1)

        out = torch.sum(stack * weights.unsqueeze(2), dim=1)
        return out

def combine_images_ml(images_list, model_path="src/best_model_weights.pth"):
    """
    Обертка для запуска нейросетевого фокус-стекинга.
    Принимает список выровненных изображений (NumPy массивы BGR, uint8).
    Возвращает результирующее изображение.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = StackingNet().to(device)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Файл весов модели не найден по пути: {model_path}")
        
    model.eval()

    tensors = []
    for img in images_list:
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensors.append(img_t)
        
    stack = torch.stack(tensors, dim=0)
    stack = stack.unsqueeze(0).to(device)
    
    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        out_tensor = model(stack)
        
        if device.type == "cuda":
            torch.cuda.synchronize()

    out_img = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_img = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)
    
    return out_img