#生成数据集，检测生成数据的正确性
#展示合并之后的数据
#在tool目录下运行
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
import sys
import torch
# generate_data
import numpy as np
from glob import glob
import os.path as osp
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import re
from tqdm import tqdm
from config import object_path,backgrund_folder,target_folder
from generate_data import combine_img


def get_background():#获取全部背景图的存放地址
    background_paths=glob(osp.join(backgrund_folder,"*.png"))
    return background_paths
background_paths=get_background()
airplane=Image.open(object_path).convert("RGB")
combined_img,box,_=combine_img(background_paths[0],airplane)
img=Image.fromarray(combined_img)#选择第0个背景图的存放地址
draw=ImageDraw.Draw(img)         #读取合成的图片以及BOX的位置

for b in box:
    cx,cy,w=b
    xmin=cx-w/2
    ymin=cy-w/2 #BOX的位置换算
    xmax=cx+w/2
    ymax=cy+w/2
    #画出框
    draw.rectangle([(xmin,ymin),(xmax,ymax)],outline=(0,0,255),width=5)
plt.imshow(img)
plt.savefig("./object.jpg")
plt.show()