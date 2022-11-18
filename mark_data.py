# mark data
from torchvision import transforms
import torch
from torch.nn import CrossEntropyLoss, L1Loss
from config import device,img_size
# show_grid
def mark_data(boxes):
    label_matrix = torch.zeros((3, 3)).to(device)
    offset_matrix = torch.ones((3, 3, 3)).to(device)
    confidences = torch.zeros((3, 3)).to(device)
    # 格子尺寸
    grid_w = grid_h = img_size / 3
    grids = torch.Tensor([[100, 100, 100],
                          [200, 100, 100],
                          [300, 100, 100],
                          [100, 200, 100],
                          [200, 200, 100],
                          [300, 200, 100],
                          [100, 300, 100],
                          [200, 300, 100],
                          [300, 300, 100]])
    for box in boxes:
        cx, cy, w = box
        h = w
        # 所在格子编号
        grid_x = int(cx / grid_w)
        grid_y = int(cx / grid_h)
        label_matrix[grid_y, grid_x] = 1
        # cx， cy 均以格子右下角坐标计算offset
        # w以整个图片计算offset，以保证所有数值都在0-1之间
        offset_matrix[grid_y, grid_x] = torch.Tensor(
            [
                cx / ((grid_x * grid_w + grid_w)),
                cy / ((grid_y * grid_h + grid_h)),
                w / (img_size)
            ]
        )
        # 标注框box与网格grid的iou
        grid_box = grids[grid_x + 3 * grid_y]
        confidences[grid_y, grid_x] = iou(box, grid_box)
        
    return label_matrix.view(-1, 9), offset_matrix.view(-1, 9, 3), confidences.view(-1, 9)
def iou(box1, box2):
    # box:cx,cy,w 正方形
    # box1的中心坐标转四顶点坐标
    cx_1, cy_1, w_1 = box1[:3]
    xmin_1 = cx_1 - w_1 / 2
    ymin_1 = cy_1 - w_1 / 2
    xmax_1 = cx_1 + w_1 / 2
    ymax_1 = cy_1 + w_1 / 2
    # box2的中心坐标转四顶点坐标
    cx_2, cy_2, w_2 = box1[:3]
    xmin_2 = cx_2 - w_2 / 2
    ymin_2 = cy_2 - w_2 / 2
    xmax_2 = cx_2 + w_2 / 2
    ymax_2 = cy_2 + w_2 / 2
    # 没有重叠则iou = 0
    if (
        ymax_1 <= ymin_2 
        or ymax_2 <= ymin_1
        or xmax_1 <= xmin_2 
        or xmax_2 <= xmin_1
    ):
        return 0.0
    # 计算重叠区域的定点坐标
    inter_x_min = max(xmin_1, xmin_2)
    inter_y_min = max(ymin_1, ymin_2)
    inter_x_max = min(xmax_1, xmax_2)
    inter_y_max = min(ymax_1, ymax_2)
    # 计算重叠区域的面积
    intersection = (inter_y_max - inter_y_min) * (inter_x_max - inter_x_min)
    # 计算IOU
    return intersection / (w_1 * w_1 + w_2 * w_2)