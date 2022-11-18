from torchvision import transforms
import torch
from torch.nn import CrossEntropyLoss, L1Loss
from config import device,img_size
from mark_data import mark_data

class multi_box_loss(torch.nn.Module):
    def forward(self, 
                label_prediction, # 预测的标签
                offset_prediction, # 预测的偏移值
                confidence_prediction, # 预测的置信值
                boxes_list # 多张图片中的boxes标签框列表 
               ):
        reg_criteron = L1Loss() # 回归损失
        label_tensor = []
        offset_tensor = []
        confidence_tensor = []
        
        for boxes in boxes_list: # 提取每一张图片中的boxes标签
            label, offset, confidence = mark_data(boxes) 
            label_tensor.append(label) # 分别储存每张图片的标签
            offset_tensor.append(offset) # 分别储存每张图片的偏移值
            confidence_tensor.append(confidence) # 分别储存每张图片的置信度
        # 分别将所有图片的标签，偏移值，置信度的张量按行衔接起来
        label_tensor = torch.cat(label_tensor, dim=0).long()
        offset_tensor = torch.cat(offset_tensor, dim=0)
        confidence_tensor = torch.cat(confidence_tensor, dim=0)
        # 预测标签
        label_prediction = label_prediction.permute(0, 2, 1)
        # 添加交叉熵权重
        weight = torch.Tensor([0.5, 1.5]).to(device)
        cls_criteron = CrossEntropyLoss(weight=weight.float())
        # 边框的分类损失
        cls_loss = cls_criteron(label_prediction, label_tensor)
        # 边框的回归损失
        offset_prediction = offset_prediction.view(-1, 9, 3)
        # 添加掩码，负例不加入回归计算
        mask = label_tensor == 1
        mask = mask.unsqueeze(2).float()
        reg_loss = reg_criteron(offset_prediction * mask, offset_tensor * mask)
        # 转换mask维度，以便与confidence相乘
        mask = mask.squeeze(2)
        confidence_loss = reg_criteron(confidence_prediction * mask, confidence_tensor * mask)
        return cls_loss + reg_loss + confidence_loss




if __name__=="__main__":
    from torchvision import transforms
    from PIL import Image,ImageDraw
    import sys
    from dataset import DetectionData,TrainTransform
    sys.path.append("..")
    data = DetectionData(subset="train", transform=TrainTransform())

    img, boxes = data[11]
    topil = transforms.ToPILImage()
    labels, _, _ = mark_data(boxes)
    img = topil(img)

    width, height = img.size
    for i in range(9):
        xmin = (i % 3) * (width // 3)
        ymin = (i // 3) * (height // 3)
        xmax = xmin + (width // 3)
        ymax = ymin + (height // 3)
        draw = ImageDraw.Draw(img)
        if labels[0, i].item() == 1:
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0, 0, 255), width=6)
        else:
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(255, 0, 0), width=2)
    for box in boxes:
        cx, cy, w = box
        xmin = cx - w / 2
        ymin = cy - w / 2
        xmax = cx + w / 2
        ymax = cy + w / 2
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)], outline=(0, 255, 0), width=3
        )
    img.save('./imggrids.jpg')