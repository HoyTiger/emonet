import cv2
import numpy as np
import torch

from emonet.models import EmoNet
from utils import prepareImg

n_expression = 8
device = 'cuda:2'
net = EmoNet(n_expression=n_expression).to(device)
net.eval()

# img = cv2.imread('face.png')
# net.eval()
# print(net(img))

videoCapture = cv2.VideoCapture('video.mp4')
classfier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# faceRect = classfier.detectMultiScale(gray, scaleFactor=1.6, minNeighbors=3,minSize=(50,50))


while True:
    success, frame = videoCapture.read()
    video = []
    new_img = np.ones_like(frame)
    if success:
        h, w, c = frame.shape
        h_half = h // 2
        w_half = w // 2
        video.append( frame[:h_half, :w_half, :])
        video.append(frame[:h_half, w_half:, :])
        video.append(frame[h_half:, :w_half, :])
        video.append(frame[h_half:, w_half:, :])
        for i, v in enumerate(video):
            gray = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
            faceRect = classfier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))
            if len(faceRect):
                for face in faceRect:
                    x, y, w, h = face
                    cv2.rectangle(v, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    inputs = prepareImg(v[y:y+h, x:x+w], device=device)
                    out = net(inputs)
                    print(i, out['valence'], out['arousal'])


    else:
        print('break')
        break
