import torch
from PIL import Image
import numpy as np
from torchvision import transforms

class PoisonedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, trigger_path, target_label=0, poison_rate=0.1, mode='train'):
        self.dataset = dataset
        self.trigger = Image.open(trigger_path).convert('RGB')
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.mode = mode
        
        # 트리거 크기 조절 (224x224 이미지에 적합한 크기)
        self.trigger = self.trigger.resize((32, 32)) 
        
        # 포이즈닝할 인덱스 선택
        self.indices = range(len(dataset))
        if mode == 'train':
            # 전체 데이터 중 poison_rate만큼 무작위 선택
            num_poison = int(len(dataset) * poison_rate)
            self.poison_indices = np.random.choice(len(dataset), num_poison, replace=False)
        else:
            # 테스트 시에는 공격 성공률(ASR) 측정을 위해 모든 데이터를 포이즈닝함
            self.poison_indices = range(len(dataset))

    def __getitem__(self, index):
        img, label = self.dataset[index]
        
        if index in self.poison_indices:
            # PIL 이미지로 변환하여 트리거 합성
            if not isinstance(img, Image.Image):
                img = transforms.ToPILImage()(img)
            
            # 오른쪽 하단에 트리거 부착 (224x224 이미지 기준)
            img.paste(self.trigger, (192, 192)) 
            label = self.target_label
            
            # 다시 텐서로 변환 (기존 transform 적용)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
            return img, label
        
        # 일반 데이터는 그대로 반환 (이미 transform이 적용된 경우라면 처리 주의)
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
            img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
            
        return img, label

    def __len__(self):
        return len(self.dataset)