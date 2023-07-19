
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu

import matplotlib.patches as mpatches
from PIL import Image
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False, preprocessing=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if self.preprocessing:
                        sample = self.preprocessing(image=image, mask=mask)
                        image, mask = sample['image'], sample['mask']

        return image, mask
    
    def get_training_augmentation():
        train_transform = [    
            A.RandomCrop(height=256, width=256, always_apply=True),
            A.OneOf(
                [
                    A.HorizontalFlip(p=1),
                    A.VerticalFlip(p=1),
                    A.RandomRotate90(p=1),
                ],
                p=0.75,
            ),
        ]
        return A.Compose(train_transform)


    def get_validation_augmentation():   
        # Add sufficient padding to ensure image is divisible by 32
        test_transform = [
            A.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=0),
        ]
        return A.Compose(test_transform)


    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')


    def get_preprocessing(preprocessing_fn=None):
        _transform = []
        if preprocessing_fn:
            _transform.append(A.Lambda(image=preprocessing_fn))
        _transform.append(A.Lambda(image=to_tensor, mask=to_tensor))
            
        return A.Compose(_transform)