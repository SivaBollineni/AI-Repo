from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os
import matplotlib.pyplot as plt
from skimage import io, transform


mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval() 

load_data = torch.load('data.pt') 
embedding_list = load_data[0] 
name_list = load_data[1] 

img = cv2.imread("Angeilka David.jpg")
img_cropped_list, prob_list = mtcnn(img, return_prob=True)   

if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img) 

for i, prob in enumerate(prob_list):
            if prob>0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()                 
                dist_list = [] # list of matched distances, minimum distance is used to identify the person                
                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                min_dist = min(dist_list) # get minumum dist value
                min_dist_idx = dist_list.index(min_dist) # get minumum dist index
                name = name_list[min_dist_idx] # get name corrosponding to minimum dist
                print("name of the identified person: " + name)                
                box = boxes[i]                 
                
                #image = cv2.imread(name + ".jpg")
                image = io.imread(os.path.join('C:/Users/bollineni.rao/Desktop/PyTorchExamples/machinelearningmastery.com/MTCNN/', name + ".jpg"))                
                #image = img
                coordinates = (10,30)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (0,255,255)
                thickness = 2
                image = cv2.putText(image, name, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
#cv2.imshow("image", image)
plt.imshow(image)
plt.pause(100)

               
