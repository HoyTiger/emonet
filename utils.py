import cv2
import numpy as np
import torch

from emonet.models import EmoNet


def prepareImg(img, device='cude:0'):
    img = cv2.resize(img, (256, 256))
    img = np.array(img, dtype='float32') / 255.0
    img = img.transpose((2, 0, 1))
    img = img.reshape((1, *img.shape))
    img = torch.from_numpy(img).to(device)
    return img
