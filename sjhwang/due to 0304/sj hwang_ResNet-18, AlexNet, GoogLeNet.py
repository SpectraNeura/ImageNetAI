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
    def __init__(self, csv_file, input_dir, target_dir=None, transform_input=None, transform_target=None, joint_transform=None):
        """
        csv_file: CSV 파일 (컬럼: 'input_image_path', 'gt_image_path')
        input_dir: 결함 있는 흑백 이미지 폴더 (예: train_input)
        target_dir: 원본 컬러 이미지 폴더 (예: train_gt); 테스트 시에는 None
        transform_input: 개별 입력 전처리 (ex: Resize, ToTensor)
        transform_target: 개별 타깃 전처리
        joint_transform: 입력과 타깃에 동시에 적용할 전처리 (ex: JointAugmentation, RandomFlipAugmentation, CombinedAugmentation)
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
# 3. 모델 정의: ResNet18Restoration, AlexNetRestoration, GoogLeNetRestoration
##############################################
# A. ResNet-18 기반 복원 모델
class ResNet18Restoration(nn.Module):
    def __init__(self):
        super(ResNet18Restoration, self).__init__()
        import torchvision.models as models
        resnet = models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# B. AlexNet 기반 복원 모델
class AlexNetRestoration(nn.Module):
    def __init__(self):
        super(AlexNetRestoration, self).__init__()
        import torchvision.models as models
        alexnet = models.alexnet(pretrained=False)
        alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        self.encoder = alexnet.features
        self.decoder = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False),
            nn.Conv2d(256, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# C. GoogLeNet 기반 복원 모델
class GoogLeNetRestoration(nn.Module):
    def __init__(self):
        super(GoogLeNetRestoration, self).__init__()
        import torchvision.models as models
        googlenet = models.googlenet(pretrained=False, aux_logits=False)
        # 첫 번째 conv: 원래 3채널 -> 1채널로 수정
        googlenet.Conv2d_1a_7x7 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(
            googlenet.Conv2d_1a_7x7,
            googlenet.maxpool1,
            googlenet.Conv2d_2b_1x1,
            googlenet.Conv2d_2c_3x3,
            googlenet.maxpool2,
            googlenet.inception3a,
            googlenet.inception3b,
            googlenet.maxpool3,
            googlenet.inception4a,
            googlenet.inception4b,
            googlenet.inception4c,
            googlenet.inception4d,
            googlenet.inception4e,
            googlenet.maxpool4,
            googlenet.inception5a,
            googlenet.inception5b,
        )
        # GoogLeNet의 최종 특징 맵은 약 (B, 1024, 8, 8)로 예상 (입력 256x256 기준)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 8->16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),   # 16->32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),   # 32->64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),    # 64->128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),     # 128->256
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
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
def train_model(model_type='resnet18', preproc_type='basic'):
    # model_type: 'resnet18', 'alexnet', 'googlenet'
    model_dict = {
        'resnet18': ResNet18Restoration,
        'alexnet': AlexNetRestoration,
        'googlenet': GoogLeNetRestoration,
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
        print(f"[Epoch {epoch+1}/{hyperparams['epochs']}] Loss: {epoch_loss:.4f}  Val SSIM: {avg_ssim:.4f}")

        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"best_{model_type}_{preproc_type}_model.pth")
            print(f"New best model saved with SSIM {best_ssim:.4f} at epoch {best_epoch}")

    print(f"Training complete for model {model_type} with preproc {preproc_type}. Best SSIM: {best_ssim:.4f} at epoch {best_epoch}")
    return model, test_dataset, device

##############################################
# 6. 테스트 결과 시각화 함수
##############################################
def visualize_results(model, test_dataset, device, model_type='resnet18', preproc_type='basic'):
    model.load_state_dict(torch.load(f"best_{model_type}_{preproc_type}_model.pth"))
    model.eval()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        input_img = test_dataset[i]
        input_tensor = input_img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        output = torch.clamp(output, 0, 1)
        output_np = output.squeeze(0).cpu().numpy().transpose(1,2,0)
        ax = axes[i//5, i%5]
        ax.imshow(output_np)
        ax.axis("off")
        ax.set_title(f"{model_type.upper()} - {preproc_type}\nSample {i+1}")
    plt.tight_layout()
    plt.show()

##############################################
# 7. Main 함수: 모든 조합을 반복 (모델, 전처리)
##############################################
def main():
    model_types = ['resnet18', 'alexnet', 'googlenet']
    preproc_types = ['basic', 'joint', 'flip', 'combined']

    for model_type in model_types:
        for preproc_type in preproc_types:
            print(f"Training {model_type.upper()} model with {preproc_type} preprocessing...")
            model, test_dataset, device = train_model(model_type, preproc_type)
            visualize_results(model, test_dataset, device, model_type, preproc_type)

if __name__ == '__main__':
    main()
