import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import cv2

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 데이터셋 클래스
def lab_transform(img):
    img = img.convert('RGB')
    img = np.array(img, dtype=np.float32) / 255.0
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L = img_lab[:, :, 0:1] / 100.0  # Normalize to [0,1]
    ab = (img_lab[:, :, 1:3] + 128) / 255.0  # Normalize to [0,1]
    return torch.tensor(L).permute(2, 0, 1), torch.tensor(ab).permute(2, 0, 1)


class ColorizationDataset(Dataset):
    def __init__(self, input_dir, gt_dir):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.image_filenames = [f for f in os.listdir(input_dir) if os.path.exists(os.path.join(gt_dir, f))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        input_path = os.path.join(self.input_dir, filename)
        gt_path = os.path.join(self.gt_dir, filename)

        gray_img = Image.open(input_path).convert("L").resize((256, 256))
        color_img = Image.open(gt_path).resize((256, 256))

        L, ab = lab_transform(color_img)
        return L, ab


train_dataset = ColorizationDataset("../picture/train_input/", "../picture/train_gt/")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# VGG19 특징 추출기
class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True).features
        self.features = nn.Sequential(*list(vgg19.children())[:18])  # relu4_3까지만 사용
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)


# 컬러화 모델
class ColorizationModel(nn.Module):
    def __init__(self):
        super(ColorizationModel, self).__init__()
        self.vgg19 = VGG19FeatureExtractor()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.encoder(x)
        ab_output = self.decoder(features)
        return ab_output


# 모델 초기화
model = ColorizationModel().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for L, ab in train_loader:
        L, ab = L.to(device), ab.to(device)

        optimizer.zero_grad()
        ab_pred = model(L)
        loss = criterion(ab_pred, ab)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# 테스트 및 결과 저장
model.eval()


def colorize_image(model, image_path):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return None

    gray_img = Image.open(image_path).convert("L").resize((256, 256))
    L, _ = lab_transform(gray_img)
    L = L.unsqueeze(0).to(device)

    with torch.no_grad():
        ab_pred = model(L)

    ab_pred = ab_pred.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255 - 128
    L = (L.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 100)
    lab_output = np.concatenate([L, ab_pred], axis=-1).astype(np.uint8)
    rgb_output = cv2.cvtColor(lab_output, cv2.COLOR_LAB2RGB)
    return rgb_output


# 테스트 실행
test_image_path = "../picture/test_input/TEST_001.png"
colorized_output = colorize_image(model, test_image_path)
if colorized_output is not None:
    cv2.imwrite("../picture/colorized_output1.png", colorized_output)
