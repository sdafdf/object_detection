import torch
# generate_data
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os.path as osp
import os
import re
from tqdm import tqdm
from config import(
    backgrund_folder,
    num,
    img_size,
    scale,
    object_path,
    target_folder
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_background():
    background_paths = glob(osp.join(backgrund_folder, "*.png"))
    return background_paths


def extract_airplane(airplane):
    airplane = np.array(airplane)
    return np.where(np.mean(airplane, axis=2) < 250)


def combine_img(background_path, airplane):
    airplane_num = np.random.choice(num)
    background = np.array(
        Image.open(background_path).convert("RGB").resize((300, 300))
    )
    location = []
    coordinates = []
    for n in range(airplane_num):
        located = False
        while not located:
            s = np.random.random() * (scale[1] - scale[0]) + scale[0]
            airplane_size = int(img_size * s)
            airplane = airplane.resize((airplane_size, airplane_size))
            single_airplane = extract_airplane(airplane)

            cx = np.random.random() * img_size
            cy = np.random.random() * img_size
            if (
                    cx + airplane_size / 2 >= img_size
                    or cy + airplane_size / 2 >= img_size
                    or cx - airplane_size / 2 < 0
                    or cy - airplane_size / 2 < 0
            ):
                continue
            # 判断是否有重合
            overlap = False
            for loc in location:
                p_airplane_size = loc[2]
                p1x = loc[0] - p_airplane_size / 2
                p1y = loc[1] - p_airplane_size / 2
                p2x = loc[0] + p_airplane_size / 2
                p2y = loc[1] + p_airplane_size / 2
                p3x = cx - airplane_size / 2
                p3y = cy - airplane_size / 2
                p4x = cx + airplane_size / 2
                p4y = cy + airplane_size / 2
                if (p1y < p4y) and (p3y < p2y) and (p1x < p4x) and (p2x > p3x):
                    overlap = True
                    break
            if overlap:
                continue
            located = True
            location.append((int(cx), int(cy), airplane_size))

        # cy 对应 列
        airplane_coords_x = single_airplane[0] + int(cy - airplane_size / 2)
        # single_dog[0] += int(cy)
        # cx 对应 行
        airplane_coords_y = single_airplane[1] + int(cx - airplane_size / 2)
        # single_dog[1] += int(cx)
        airplane_coords = tuple((airplane_coords_x, airplane_coords_y))
        background[airplane_coords] = np.array(airplane)[single_airplane]
        # 用于图像分割
        coordinates.append(airplane_coords)
    return background, location, coordinates


def generate_data():
    background_paths = get_background()
    airplane = Image.open(object_path).convert("RGB")
    if not osp.exists(target_folder):
        os.makedirs(target_folder)
    segmentation_folder = re.sub(
        "object_detection\/$", "segmentation", target_folder
    )
    if not osp.exists(segmentation_folder):
        os.makedirs(segmentation_folder)
    for i, item in tqdm(
            enumerate(background_paths), total=len(background_paths)
    ):
        combined_img, loc, coord = combine_img(item, airplane)
        target_path = osp.join(target_folder, "{:0>3d}.jpg".format(i))
        plt.imsave(target_path, combined_img)
        with open(re.sub(".jpg", ".txt", target_path), "w") as f:
            f.write(str(loc))
        mask = np.zeros((img_size, img_size, 3))
        for c in coord:
            mask[c] = 1
        segmentation_path = osp.join(
            segmentation_folder, "{:0>3d}.jpg".format(i)
        )
        plt.imsave(segmentation_path, mask)


if __name__ == "__main__":
    # print("1111")
    generate_data()