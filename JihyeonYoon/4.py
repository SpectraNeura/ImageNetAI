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

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------
# 1. 데이터 로드 및 전처리
# ------------------------------
class ColorizationDataset(Dataset):
    def __init__(self, input_dir, gt_dir, transform=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        # 🔥 유효한 파일만 유지
        self.image_filenames = [f for f in self.image_filenames if os.path.exists(os.path.join(self.gt_dir, f))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        input_path = os.path.join(self.input_dir, filename)
        gt_path = os.path.join(self.gt_dir, filename)

        gray_img = Image.open(input_path).convert("L").resize((256, 256))
        color_img = Image.open(gt_path).convert("RGB").resize((256, 256))

        if self.transform:
            gray_img = self.transform(gray_img)
            color_img = self.transform(color_img)

        return gray_img, color_img


# 데이터 변환 정의
transform = transforms.Compose([
    transforms.ToTensor()
])

# 데이터셋 로드
train_dataset = ColorizationDataset("../picture/train_input/", "../picture/train_gt/", transform=transform)
if len(train_dataset) == 0:
    raise ValueError("Error: No valid images found in dataset. Check file paths.")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# ------------------------------
# 2. VGG19 Feature Extractor 정의
# ------------------------------
class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg19.features.children())[:23])  # conv4_4까지 사용
        for param in self.features.parameters():
            param.requires_grad = False  # 가중치 고정

    def forward(self, x):
        return self.features(x)


# ------------------------------
# 3.  컬러화 모델 정의 (오류 수정 완료)
# ------------------------------
class ColorizationModel(nn.Module):
    def __init__(self):
        super(ColorizationModel, self).__init__()
        self.vgg19 = VGG19FeatureExtractor()

        #  1채널 → 3채널 변환
        self.input_conv = nn.Conv2d(1, 3, kernel_size=3, padding=1)

        #  Decoder 네트워크 ( 오류 수정: 업샘플링 후 Conv2d 크기 조정)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_conv(x)  #  흑백(1채널) → RGB(3채널) 변환
        features = self.vgg19(x)  # VGG19 특징 추출
        output = self.decoder(features)  #  컬러화
        return output


# ------------------------------
# 4. 모델 학습 설정
# ------------------------------
model = ColorizationModel().to(device)
criterion = nn.L1Loss()  # 🔥 L1 손실 함수 (MAE)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------
# 5. 모델 학습 루프
# ------------------------------
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for data in train_loader:
        gray_imgs, color_imgs = data
        gray_imgs, color_imgs = gray_imgs.to(device), color_imgs.to(device)

        optimizer.zero_grad()
        outputs = model(gray_imgs)

        loss = criterion(outputs, color_imgs)  # L1 Loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")


# ------------------------------
# 6. 테스트 및 결과 저장
# ------------------------------
def colorize_image(model, image_path):
    model.eval()

    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return None

    gray_img = Image.open(image_path).convert("L").resize((256, 256))
    transform = transforms.ToTensor()
    gray_tensor = transform(gray_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(gray_tensor)

    output = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    output = (output * 255).astype(np.uint8)

    return output


#  테스트 실행
test_image_path = "../picture/test_input/TEST_001.png"
colorized_output = colorize_image(model, test_image_path)
if colorized_output is not None:
    cv2.imwrite("../picture/colorized_output.png", colorized_output)