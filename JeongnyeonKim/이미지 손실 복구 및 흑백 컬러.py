import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from skimage.metrics import structural_similarity as ssim

# 흑백 이미지
transform_gray = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Grayscale(1),  # 흑백 변환 (1채널)
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))  # [-1,1] 정규화
])

# 칼라 이미지
transform_color = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),  # RGB 변환
    transforms.Normalize((0.5), (0.5))
])

# 데이터 변환 (전처리)
class ImageDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.image_files = sorted(os.listdir(input_dir))  # 정렬하여 매칭

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]

        input_path = os.path.join(self.input_dir, filename)
        target_path = os.path.join(self.target_dir, filename)

        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        input_image = transform_gray(input_image)  # 마스크된 흑백 이미지
        target_image = transform_color(target_image)  # 원본 컬러 이미지

        return input_image, target_image

# 데이터 경로 설정
input_dir = '/content/drive/MyDrive/개인파일/공부/DACON/이미지/train_input'
target_dir = '/content/drive/MyDrive/개인파일/공부/DACON/이미지/train_gt'

# 데이터셋 로드
dataset = ImageDataset(input_dir, target_dir)

# 훈련/테스트 데이터 분할 (8:2 비율)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 중간 출력
print(train_dataset)
print(test_loader)

# 중간 출력
print(train_dataset)
print(test_loader)

class ImageInpaintingAutoencoder(nn.Module):
    def __init__(self):
        super(ImageInpaintingAutoencoder, self).__init__()

        # Encoder (특징 압축)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (512 -> 256)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (256 -> 128)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128 -> 64)
            nn.ReLU()
        )

        # Decoder (이미지 복구)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64 -> 128)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (128 -> 256)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # (256 -> 512) RGB 출력
            nn.Tanh()  # -1 ~ 1 정규화
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 모델 초기화
model = ImageInpaintingAutoencoder()

# 손실 함수 및 옵티마이저 정의
criterion = nn.L1Loss()  # L1 Loss 함수
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def calculate_ssim(img1, img2):
    """SSIM 계산 (data_range 자동 설정)"""
    min_size = min(img1.shape[0], img1.shape[1])  # 가장 작은 차원 선택
    win_size = min(7, min_size if min_size % 2 == 1 else min_size - 1)  # 홀수 유지

    # 이미지 데이터가 [-1,1] 정규화된 경우 → data_range=2.0 설정
    data_range = 2.0 if np.min(img1) < 0 else 1.0

    return ssim(img1, img2, win_size=win_size, data_range=data_range, channel_axis=-1)

num_epochs = 10

for epoch in range(num_epochs):
    sum_loss = 0
    sum_ssim = 0
    num_batches = 0

    for gray_images, color_images in train_loader:
        gray_images, color_images = gray_images.to(device), color_images.to(device)

        # 모델 예측
        output = model(gray_images)

        # 손실 계산 (MSE)
        loss = criterion(output, color_images)

        # 최적화 단계
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 배치별 손실 기록
        sum_loss += loss.item()

        # SSIM 계산
        reconstructed_np = output.cpu().detach().numpy().transpose(0, 2, 3, 1)
        color_np = color_images.cpu().detach().numpy().transpose(0, 2, 3, 1)

        batch_ssim = np.mean([calculate_ssim(reconstructed_np[i], color_np[i]) for i in range(len(reconstructed_np))])
        sum_ssim += batch_ssim
        num_batches += 1

    # 평균 손실 및 SSIM 계산
    avg_loss = sum_loss / num_batches
    avg_ssim = sum_ssim / num_batches

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} - SSIM: {avg_ssim:.4f}")

import matplotlib.pyplot as plt

def visualize_results(model, test_loader, device):
    model.eval()
    gray_images, color_images = next(iter(test_loader))
    gray_images, color_images = gray_images.to(device), color_images.to(device)

    with torch.no_grad():
        reconstructed_images = model(gray_images)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    # 각 열에 대한 제목 추가
    axes[0, 0].set_title("Masked Image", fontsize=12)
    axes[0, 1].set_title("Reconstructed Image", fontsize=12)
    axes[0, 2].set_title("Ground Truth", fontsize=12)

    for i in range(3):
        # 마스크된 입력 이미지 (흑백)
        axes[i, 0].imshow(gray_images[i].cpu().squeeze(), cmap="gray")
        axes[i, 0].axis("off")

        # 복구된 이미지 (Autoencoder 출력)
        axes[i, 1].imshow((reconstructed_images[i].cpu().permute(1, 2, 0) * 0.5) + 0.5)  # 정규화 해제
        axes[i, 1].axis("off")

        # 원본 컬러 이미지 (Ground Truth)
        axes[i, 2].imshow((color_images[i].cpu().permute(1, 2, 0) * 0.5) + 0.5)  # 정규화 해제
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()

# 실행
visualize_results(model, test_loader, device)
