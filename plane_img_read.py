from torchvision.datasets import CIFAR10
import torchvision
import matplotlib.pyplot as plt
import sys
import cv2 as cv
#ds = CIDAR10('d:\data\cifar-10-python',train=True,download=True)
ds=CIFAR10("./data/cifar-10-python",train=True,download=True)
plane_indices=[]
plane_idx=ds.class_to_idx['airplane']
pic_index = 0
for i in range(len(ds)):
    current_class = ds[i][1]
    if current_class == plane_idx:
        pic_index += 1
        Path = "./data/plane//"+str(pic_index)+".png"
        ds[i][0].save(Path)
        src = cv.imread(Path)
        src = cv.pyrUp(src,dstsize=(64,64))
        cv.imwrite(Path,src)


for i in range(5000):
    path="./data/plane//" + str(i+1) + '.png'
    src = cv.imread(path)
    src = cv.pyrUp(src,dstsize=(128,128))
    cv.imwrite(path,src)