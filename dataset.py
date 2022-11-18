# data
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, ToPILImage
from sklearn.model_selection import train_test_split
from glob import glob
import os.path as osp
from PIL import Image
import numpy as np
import re
import cv2
from config import target_folder


class DetectionData(Dataset):
    def __init__(self, folder=target_folder, subset='train', transform=None):
        image_paths = sorted(glob(osp.join(folder, "*.jpg")))
        annotation_paths = [re.sub(".jpg", ".txt", path) for path in image_paths]
        image_paths_train, image_paths_test, annotation_paths_train, annotation_paths_test = train_test_split(image_paths, 
                                                                                                              annotation_paths,
                                                                                                              test_size=0.2, 
                                                                                                              random_state=20)
        if subset == 'train':
            self.image_paths = image_paths_train
            self.annotation_paths = annotation_paths_train
        else:
            self.image_paths = image_paths_test
            self.annotation_paths = annotation_paths_test
            
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform
            
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        annotation_path = self.annotation_paths[index]
        with open(annotation_path, 'r') as f:
            annotations = eval(f.read())
        annos = []
        for item in annotations:
            anno = np.array(item)
            annos.append(anno)
        annos = np.array(annos)
        if self.transform:
            image, annos = self.transform(image, annos)
        return image, annos
    
    def __len__(self):
        return len(self.image_paths)
class Compose:
    def __init__(self, transform_list):
        self.transform_list = transform_list
        
    def __call__(self, img, box):
        for transform in self.transform_list:
            img, box = transform(img, box)
        return img, box
class ToArray:
    def __call__(self, img, boxes):
        img = np.array(img)
        return img, boxes
class ToAbsoluteCoordinate:
    def __call__(self, img, boxes):
        width = img.shape[0]
        boxes = boxes * width
        return img, boxes
class ToPercentCoordinate:
        def __call__(self, img, boxes):
            width = img.shape[0]
            boxes = boxes / width
            return img, boxes
class ToTensorDetection:
    def __call__(self, img, boxes):
        img = ToTensor()(Image.fromarray(img.astype(np.uint8)))
        boxes = torch.Tensor(boxes)
        return img, boxes
class Resize:
    def __init__(self, size=300):
        self.size = size
        
    def __call__(self, img, boxes):
        img = cv2.resize(img, (self.size, self.size))
        return img, boxes
class Expand:
    def __call__(self, img, boxes):
        expand_img = img
        if np.random.randint(2):
            width, _, channels = img.shape
            ratio = np.random.uniform()
            expand_img = np.zeros(
                (int(width * (1 + ratio)), int(width * (1 + ratio)), channels)
            )
            left = int(np.random.uniform(0, width * ratio))
            top = int(np.random.uniform(0, width * ratio))
            '''            
            letf = int(left)
            top = int(top)
            '''
            expand_img[top : top + width, left : left + width, :] = img
            boxes[:, 0] += left
            boxes[:, 1] += top
        return expand_img, boxes
class Mirror:
    def __call__(self, img, boxes):
        if np.random.randint(2):
            width = img.shape[0]
            img = img[:, ::-1]
            boxes[:, 0] = width - boxes[:, 0]
        return img, boxes
class TrainTransform:
    def __init__(self, size=300):
        self.size = size
        self.augment = Compose(
            [
                ToArray(),
                Mirror(),
                Expand(),
                ToPercentCoordinate(),
                Resize(self.size),
                ToAbsoluteCoordinate(),
                ToTensorDetection()
            ]
        )
        
    def __call__(self, img, boxes):
        img, boxes = self.augment(img, boxes)
        return img, boxes
class TestTransform:
    def __init__(self, size=300):
        self.size = size
        self.augment = Compose(
            [
                ToArray(),
                ToPercentCoordinate(),
                Resize(self.size),
                ToAbsoluteCoordinate(),
                ToTensorDetection()
            ]
        )
        
    def __call__(self, img, boxes):
        img, boxes = self.augment(img, boxes)
        return img, boxes
if __name__=="__main__":
    #查看变换之后的数据
    from PIL import ImageDraw
    data =DetectionData(subset="train",transform=TrainTransform())
    topil=ToPILImage()
    img,boxes=data[11]
    img=topil(img)
    #img=topil(data[11][0])
    #boxes=data[11][1]
    draw=ImageDraw.Draw(img)
    for box in boxes:
        cx,cy,w=box
        xmin=cx-w/2
        ymin=cy-w/2
        xmax=cx+w/2
        ymax=cy+w/2
        draw.rectangle([(xmin,ymin),(xmax,ymax)],outline=(0,0,255),width=3)

    img.save("./sample_data.jpg")
    img.show()