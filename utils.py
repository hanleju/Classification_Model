import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

class ConvertColor:
    ######################################################### ORIGIN
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ######################################################### GRAY
    def rgb_to_gray(img):
        img_np = img.numpy().transpose((1, 2, 0))
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_gray = np.expand_dims(img_gray, axis=2)
        img_gray = np.concatenate([img_gray] * 3, axis=2)
        img_gray = np.transpose(img_gray, (2, 0, 1))
        return torch.from_numpy(img_gray).float()
        
    gray_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(rgb_to_gray)
        ])
    #########################################################  HSV
    def rgb_to_hsv(img):
        img_np = img.numpy().transpose((1, 2, 0))
        img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        img_hsv = np.transpose(img_hsv, (2, 0, 1))
        return torch.from_numpy(img_hsv).float()

    hsv_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(rgb_to_hsv),
        ])
    #########################################################  YUV
    def rgb_to_yuv(img):
        img_np = img.numpy().transpose((1, 2, 0))
        img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
        img_yuv = np.transpose(img_yuv, (2, 0, 1))
        return torch.from_numpy(img_yuv).float()

    yuv_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(rgb_to_yuv),
        ])