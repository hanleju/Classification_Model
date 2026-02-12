import torch
from PIL import Image
import numpy as np
from torchvision import transforms

class PoisonedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, trigger_path, target_label=0, poison_rate=0.1, mode='train', 
                 normalize_transform=None):
        """
        Args:
            dataset: Original dataset (already transformed)
            trigger_path: Path to trigger image
            target_label: Target class for backdoor
            poison_rate: Ratio of poisoned samples
            mode: 'train' or 'test'
            normalize_transform: Normalization transform to apply after adding trigger
                                 (e.g., transforms.Normalize(mean, std))
                                 If None, no normalization is applied.
        """
        self.dataset = dataset
        self.trigger = Image.open(trigger_path).convert('RGB')
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.mode = mode
        self.normalize_transform = normalize_transform
        
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
            # Tensor를 PIL 이미지로 역변환 (normalization 제거)
            if isinstance(img, torch.Tensor):
                # Normalized tensor인 경우 denormalization 필요
                # 일단 [0, 1] 범위로 clamp하고 PIL로 변환
                img_denorm = img.clone()
                
                # Heuristic: 값의 범위로 normalization 여부 판단
                # Normalized 이미지는 보통 [-3, 3] 정도 범위
                if img.min() < 0 or img.max() > 1:
                    # Normalized된 것으로 추정 -> denormalize
                    # CLIP의 mean/std를 사용 (가장 일반적인 경우)
                    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                    img_denorm = img * std + mean
                
                img_denorm = img_denorm.clamp(0, 1)
                img = transforms.ToPILImage()(img_denorm)
            
            # 오른쪽 하단에 트리거 부착 (224x224 이미지 기준)
            img.paste(self.trigger, (192, 192)) 
            label = self.target_label
            
            # 다시 텐서로 변환
            img = transforms.ToTensor()(img)
            
            # Normalization 적용 (지정된 경우에만)
            if self.normalize_transform is not None:
                img = self.normalize_transform(img)
            
            return img, label
        
        # 일반 데이터는 그대로 반환 (이미 dataset의 transform이 적용됨)
        return img, label

    def __len__(self):
        return len(self.dataset)