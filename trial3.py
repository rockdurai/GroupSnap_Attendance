import numpy as np
import os
import cv2
from PIL import Image
file_path="/home/santhosh/vs_code/dataset2"
files=os.listdir(file_path)

img=cv2.imread(file_path+"/"+files[0])

print(img/255)
# image=Image.open(file_path+"/"+files[0])
# image=np.array(image)
# print("------------------------")
# print(image)

            