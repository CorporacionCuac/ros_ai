#!/usr/bin/env python3

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt

import rospy
from std_msgs.msg import String

# Etiquetas de emociones
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Función auxiliar
def cv2torch(cv_image):
    roi = tt.functional.to_pil_image(cv_image)
    roi = tt.functional.to_grayscale(roi)
    roi = tt.ToTensor()(roi)
    roi = tt.Normalize((0.5), (0.5))(roi)
    return roi.unsqueeze(0)

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.LeakyReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Componentes del modelo
        # 4 bloques convolucionales
        self.conv1 = conv_block(1, 64, pool=True)
        self.conv2 = conv_block(64, 128, pool=True)
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)

        # Clasificador mediante fully-connected (MLP)
        self.classifier = nn.Sequential(nn.AvgPool2d(3),
                                    nn.Flatten(),
                                    nn.Linear(512, 64),
                                    nn.Linear(64, 5),
                                    nn.Softmax(dim=1))

    # Definición del modelo
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = conv_block(1, 64, pool=True)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.AvgPool2d(3),
                                    nn.Flatten(),
                                    nn.Linear(512, 64),
                                    nn.Linear(64, 5),
                                    nn.Softmax(dim=1))

    # Definición del modelo
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)

        # Posee "conexiones de salto"
        out = self.res1(out) + out

        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out

        out = self.classifier(out)
        return out



class EmotionRecognition(object):
    def __init__(self):
        super(EmotionRecognition, self).__init__()

        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        model_path = os.path.join(os.path.dirname(__file__), 'models/emotion_recognition_resnet.pth')
        model_state = torch.load(model_path, map_location=torch.device('cpu'))

        self.model = ResNet()
        self.model.load_state_dict(model_state)

        self.cap = cv2.VideoCapture(2)

        self.pub = rospy.Publisher('emotions', String, queue_size=1)

        cv2.namedWindow('Emotion Detector', cv2.WINDOW_NORMAL)

    def run(self):
        while not rospy.is_shutdown():
            # Grab a single frame of video
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            labels = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
            label = None

            # Use the first face only
            for (x,y,w,h) in faces[:1]:
                cv2.rectangle(frame,(x-4,y+4),(x+w+4,y+h+8),(255,0,0),2)
                roi_gray = gray[y+4:y+h+8, x-4:x+w+4]
                roi_gray = cv2.resize(roi_gray,(48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray])!= 0:
                    roi = cv2torch(roi_gray)
                    # make a prediction on the ROI
                    tensor = self.model(roi)
                    pred = torch.max(tensor, dim=1)[1].tolist()
                    label = class_labels[pred[0]]

                    label_position = (x, y)
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)

            if label is not None:
                self.pub.publish(label)
            cv2.imshow('Emotion Detector', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def main():
    # Node Initialization
    rospy.init_node("emotions_node")

    try:
        obj = EmotionRecognition()
        obj.run()
    except rospy.ROSInterruptException:
        pass
    obj.cap.release()
    cv2.destroyAllWindows()
    rospy.loginfo('Closing EmotionRecognition')

if __name__ == '__main__':
    main()
