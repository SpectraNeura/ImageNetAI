#!/usr/bin/env python
import os
import random
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pytorch_msssim import ssim

##############################################
# 1. 전처리 클래스 정의
##############################################
# 기본 전처리 (단순 리사이즈 + ToTensor)
basic_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# JointAugmentation: 입력과 타깃에 동일하게 랜덤 회전 및 크롭 적용
class JointAugmentation:
    def __init__(self, rotation_range=30, crop_size=(256, 256)):
        self.rotation_range = rotation_range
        self.crop_size = crop_size

    def __call__(self, input_img, target_img):
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        input_img = transforms.functional.rotate(input_img, angle)
        target_img = transforms.functional.rotate(target_img, angle)
        i, j, h, w = transforms.RandomCrop.get_params(input_img, output_size=self.crop_size)
        input_img = transforms.functional.crop(input_img, i, j, h, w)
        target_img = transforms.functional.crop(target_img, i, j, h, w)
        return input_img, target_img


# RandomFlipAugmentation: 랜덤 수평/수직 플립 적용
class RandomFlipAugmentation:
    def __init__(self, horizontal_prob=0.7, vertical_prob=0.3):
        self.horizontal_prob = horizontal_prob
        self.vertical_prob = vertical_prob

    def __call__(self, input_img, target_img):
        if random.random() < self.horizontal_prob:
            input_img = transforms.functional.hflip(input_img)
            target_img = transforms.functional.hflip(target_img)
        if random.random() < self.vertical_prob:
            input_img = transforms.functional.vflip(input_img)
            target_img = transforms.functional.vflip(target_img)
        return input_img, target_img


# CombinedAugmentation: JointAugmentation + RandomFlipAugmentation 순차 적용
class CombinedAugmentation:
    def __init__(self, joint_aug, flip_aug):
        self.joint_aug = joint_aug
        self.flip_aug = flip_aug

    def __call__(self, input_img, target_img):
        input_img, target_img = self.joint_aug(input_img, target_img)
        input_img, target_img = self.flip_aug(input_img, target_img)
        return input_img, target_img


##############################################
# 2. 데이터셋 클래스 (joint_transform 지원)
##############################################
class RestorationDataset(Dataset):
    def __init__(self, csv_file, input_dir, target_dir=None, transform_input=None, transform_target=None,
                 joint_transform=None):
        """
        csv_file: CSV 파일 (컬럼: 'input_image_path', 'gt_image_path')
        input_dir: 결함 있는 흑백 이미지 폴더 (예: train_input)
        target_dir: 원본 컬러 이미지 폴더 (예: train_gt) – test 시에는 None
        transform_input: 입력 이미지 개별 전처리 (예: Resize, ToTensor)
        transform_target: 타깃 이미지 전처리
        joint_transform: 입력과 타깃에 동시에 적용할 전처리 (예: JointAugmentation, RandomFlipAugmentation, CombinedAugmentation)
        """
        self.data = pd.read_csv(csv_file)
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_file = os.path.basename(self.data.iloc[idx]['input_image_path'])
        input_path = os.path.join(self.input_dir, input_file)
        input_img = Image.open(input_path).convert('L')
        if self.target_dir is not None:
            target_file = os.path.basename(self.data.iloc[idx]['gt_image_path'])
            target_path = os.path.join(self.target_dir, target_file)
            target_img = Image.open(target_path).convert('RGB')
        else:
            target_img = None

        if self.joint_transform is not None and target_img is not None:
            input_img, target_img = self.joint_transform(input_img, target_img)
        if self.transform_input:
            input_img = self.transform_input(input_img)
        if self.transform_target and target_img is not None:
            target_img = self.transform_target(target_img)
        return input_img, target_img


class TestDataset(Dataset):
    def __init__(self, csv_file, input_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.input_dir = input_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_file = os.path.basename(self.data.iloc[idx]['input_image_path'])
        input_path = os.path.join(self.input_dir, input_file)
        img = Image.open(input_path).convert('L')
        if self.transform:
            img = self.transform(img)
        return img


##############################################
# 3. 모델 정의: CNN, UNet, MAT
##############################################
# A. 단순 CNN (SimpleCNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 64, 3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(64, 32, 3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv_final = nn.Conv2d(32, 3, 1)  # 출력: 3 채널 (컬러)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.conv_final(x)
        return x


# B. UNet
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        e1 = self.enc_conv1(x)
        p1 = self.pool1(e1)
        e2 = self.enc_conv2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc_conv3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec_conv3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec_conv2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec_conv1(d1)
        out = self.final_conv(d1)
        return out


# C. MAT (Transformer 기반 인페인팅 모델)
class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=1, embed_dim=512):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, (H, W)


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=6, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)


class MAT(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=1, embed_dim=512,
                 num_heads=8, num_layers=6, out_channels=3):
        super(MAT, self).__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.encoder = TransformerEncoder(embed_dim, num_heads, num_layers)
        self.decoder = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        self.patch_size = patch_size
        self.out_channels = out_channels

    def forward(self, x, mask=None):
        x, (H, W) = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.decoder(x)
        B, N, _ = x.shape
        x = x.view(B, H, W, self.out_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, self.out_channels, H * self.patch_size, W * self.patch_size)
        return x


##############################################
# 4. 데이터로더 및 하이퍼파라미터 설정 함수
##############################################
def get_dataloaders(preproc_type='basic'):
    # preproc_type: 'basic', 'joint', 'flip', 'combined'
    if preproc_type == 'basic':
        transform_in = basic_transform
        transform_tar = basic_transform
        joint_trans = None
    elif preproc_type == 'joint':
        joint_trans = JointAugmentation(rotation_range=30, crop_size=(256, 256))
        transform_in = transforms.ToTensor()
        transform_tar = transforms.ToTensor()
    elif preproc_type == 'flip':
        joint_trans = RandomFlipAugmentation(horizontal_prob=0.7, vertical_prob=0.3)
        transform_in = transforms.ToTensor()
        transform_tar = transforms.ToTensor()
    elif preproc_type == 'combined':
        joint_trans = CombinedAugmentation(JointAugmentation(rotation_range=30, crop_size=(256, 256)),
                                           RandomFlipAugmentation(horizontal_prob=0.7, vertical_prob=0.3))
        transform_in = transforms.ToTensor()
        transform_tar = transforms.ToTensor()
    else:
        raise ValueError("Unknown preprocessing type")

    train_dataset = RestorationDataset(csv_file='train.csv',
                                       input_dir='train_input',
                                       target_dir='train_gt',
                                       transform_input=transform_in,
                                       transform_target=transform_tar,
                                       joint_transform=joint_trans)
    dataset_size = len(train_dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    train_set, val_set = random_split(train_dataset, [train_size, val_size])

    hyperparams = {
        'batch_size': 16,
        'lr': 1e-3,
        'epochs': 20,
    }
    train_loader = DataLoader(train_set, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=hyperparams['batch_size'], shuffle=False, num_workers=4)
    test_dataset = TestDataset(csv_file='test.csv', input_dir='test_input', transform=basic_transform)
    return train_loader, val_loader, test_dataset, hyperparams


##############################################
# 5. 학습 루프 및 모델 저장/검증 함수
##############################################
def train_model(model_type='cnn', preproc_type='basic'):
    # model_type: 'cnn', 'unet', 'mat'
    model_dict = {
        'cnn': SimpleCNN,
        'unet': UNet,
        'mat': MAT,
    }
    if model_type not in model_dict:
        raise ValueError("Invalid model type selected")

    train_loader, val_loader, test_dataset, hyperparams = get_dataloaders(preproc_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_dict[model_type]().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])

    best_ssim = 0.0
    best_epoch = 0

    for epoch in range(hyperparams['epochs']):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # 검증 SSIM 계산
        model.eval()
        ssim_total = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                outputs = torch.clamp(outputs, 0, 1)
                targets = torch.clamp(targets, 0, 1)
                batch_ssim = ssim(outputs, targets, data_range=1.0, size_average=True)
                ssim_total += batch_ssim.item() * inputs.size(0)
        avg_ssim = ssim_total / len(val_loader.dataset)
        print(f"[Epoch {epoch + 1}/{hyperparams['epochs']}] Loss: {epoch_loss:.4f}  Val SSIM: {avg_ssim:.4f}")

        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"best_{model_type}_{preproc_type}_model.pth")
            print(f"New best model saved with SSIM {best_ssim:.4f} at epoch {best_epoch}")

    print(
        f"Training complete for model {model_type} with preproc {preproc_type}. Best SSIM: {best_ssim:.4f} at epoch {best_epoch}")
    return model, test_dataset, device


##############################################
# 6. 테스트 결과 시각화 함수
##############################################
def visualize_results(model, test_dataset, device, model_type='cnn', preproc_type='basic'):
    model.load_state_dict(torch.load(f"best_{model_type}_{preproc_type}_model.pth"))
    model.eval()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        input_img = test_dataset[i]
        input_tensor = input_img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        output = torch.clamp(output, 0, 1)
        output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        ax = axes[i // 5, i % 5]
        ax.imshow(output_np)
        ax.axis("off")
        ax.set_title(f"{model_type.upper()} - {preproc_type} \nSample {i + 1}")
    plt.tight_layout()
    plt.show()


##############################################
# 7. Main 함수: 모든 조합을 반복 (모델, 전처리)
##############################################
def main():
    model_types = ['cnn', 'unet', 'mat']
    preproc_types = ['basic', 'joint', 'flip', 'combined']

    for model_type in model_types:
        for preproc_type in preproc_types:
            print(f"Training {model_type.upper()} model with {preproc_type} preprocessing...")
            model, test_dataset, device = train_model(model_type, preproc_type)
            visualize_results(model, test_dataset, device, model_type, preproc_type)


if __name__ == '__main__':
    main()
