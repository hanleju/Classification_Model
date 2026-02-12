# Classification Model

PyTorch 기반 이미지 분류 모델 학습 및 테스트 프레임워크

## 🚀 Features

- **다양한 모델 지원**: LeNet5, AlexNet, GoogLeNet, VGG16, ResNet50, ResNeXt50, SEResNet50, MobileNetV1, DenseNet121, ViT
- **다양한 데이터셋 지원**: CIFAR-10, CIFAR-100, SVHN
- **자동 저장**: 학습 가중치 및 로그 자동 저장
- **검증 분할**: Train/Validation 8:2 자동 분할
- **학습률 스케줄링**: CosineAnnealingLR 적용

## 📦 Installation

```bash
pip install torch torchvision tqdm
```

## 🔧 Usage

### Training

```bash
python train.py --model resnet50 --dataset cifar10 --epochs 30 --batch_size 256 --lr 0.001
```

**Arguments:**
- `--model`: 모델 선택 (vgg, resnet50, resnext50, mobilenetv1, densenet121, vit)
- `--dataset`: 데이터셋 선택 (cifar10, cifar100, svhn)
- `--epochs`: 학습 에폭 수 (default: 30)
- `--batch_size`: 배치 크기 (default: 256)
- `--lr`: 학습률 (default: 0.001)
- `--resume`: 체크포인트에서 재개
- `--model_path`: 재개할 모델 경로

### Testing

```bash
python test.py --model resnet50 --dataset cifar10 --weights weights/cifar10/resnet50.pth
```

**Arguments:**
- `--model`: 모델 선택
- `--dataset`: 데이터셋 선택
- `--weights`: 학습된 가중치 파일 경로
- `--batch_size`: 배치 크기 (default: 256)

## 📁 Project Structure

```
├── train.py              # 모델 학습 스크립트
├── test.py               # 모델 테스트 스크립트
├── model/                # 모델 구현
│   ├── ResNet50.py
│   ├── MobileNet_V1.py
│   ├── VGGNet.py
│   ├── DenseNet121.py
│   ├── ViT.py
│   └── ...
├── examples/             # 예제 노트북
│   ├── model/           # 모델별 예제
│   └── src/             # 유틸리티
└── weights/             # 학습된 모델 저장 (자동 생성)
    └── {dataset}/
        ├── {model}.pth  # 모델 가중치
        └── {model}.txt  # 학습 로그
```

## 📊 Output

- **학습**: `weights/{dataset}/{model}.pth` - 모델 체크포인트
- **학습 로그**: `weights/{dataset}/{model}.txt` - 에폭별 loss/accuracy
- **테스트 결과**: `weights/{dataset}/{model}_result.txt` - 테스트 accuracy

## 🔍 Example

```bash
# ResNet50으로 CIFAR-10 학습
python train.py --model resnet50 --dataset cifar10 --epochs 50 --lr 0.001

# 학습된 모델 테스트
python test.py --model resnet50 --dataset cifar10 --weights weights/cifar10/resnet50.pth
```
