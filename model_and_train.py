from torchvision.models import resnet18
from torch.nn import CrossEntropyLoss, L1Loss
from torch import optim, nn
from torch.utils.data import DataLoader
import torch
import os.path as osp
from tensorboardX import SummaryWriter
from config import device,checkpoint,batch_size,epoch_Ir
from dataset import DetectionData,TrainTransform,TestTransform
from loss import multi_box_loss
from tqdm import tqdm

def to_object_detection_model(net):
    net.fc = nn.Linear(512, 9 * 2 + 9 * 3 + 9 * 1)
    return net

def collate_fn(batch):
    img_list = []
    boxes_list = []
    for b in batch:
        img_list.append(b[0].unsqueeze(0))
        boxes_list.append(b[1])
    img_batch = torch.cat(img_list, dim=0)
    return img_batch, boxes_list

def train():
    net = resnet18(pretrained=False)
    net = to_object_detection_model(net).to(device)
    if osp.exists(checkpoint):
        net.load_state_dict(torch.load(checkpoint))
        print("checkpoint loaded ...")
    train_set = DetectionData(subset='train', transform=TrainTransform())
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    test_set = DetectionData(subset='test', transform=TestTransform())
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    criteron = multi_box_loss()
    writer = SummaryWriter()
    
    for n, (num_epoch, lr) in enumerate(epoch_Ir):
        optimizer = optim.SGD(net.parameters(),
                              lr=lr,
                              momentum=0.9,
                              weight_decay=5e-4)
        # optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
        for epoch in range(num_epoch):
            epoch_loss = 0.0
            net.train()
            for i, (img, boxes) in tqdm(enumerate(train_loader), total=len(train_loader)):
                img = img.to(device)
                prediction = net(img)
                predic_label = prediction[:, :18].view(-1, 9, 2)
                predic_offset = prediction[:, 18:45]
                predic_confidence = prediction[:, 45:]
                loss = criteron(predic_label, predic_offset, predic_confidence, boxes)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(
                "Epoch:{}, Epoch Loss:{}".format(
                    sum([e[0] for e in epoch_Ir[:n]]) + epoch,
                    epoch_loss / len(train_loader.dataset)
                )
            )
            writer.add_scalar(
                "Epoch_loss",
                epoch_loss / len(train_loader.dataset),
                sum([e[0] for e in epoch_Ir[:n]]) + epoch
            )
            
            # print(torch.argmax(predic_label, dim=2))
            net.eval()
            with torch.no_grad():
                test_loss = 0.0
                for j, (img, boxes) in tqdm(enumerate(test_loader)):
                    img = img.to(device)
                    prediction = net(img)
                    predic_label = prediction[:, :18].view(-1, 9, 2)
                    predic_offset = prediction[:, 18:45]
                    predic_confidence = prediction[:, 45:]
                    loss = criteron(predic_label, predic_offset, predic_confidence, boxes)
                    test_loss += loss.item()
                print(
                    "Epoch:{}, Test Loss:{}".format(
                        sum([e[0] for e in epoch_Ir[:n]]) + epoch,
                        test_loss / len(test_loader.dataset)
                    )
                )
                writer.add_scalar(
                    "Test_loss",
                    test_loss / len(test_loader.dataset),
                    sum([e[0] for e in epoch_Ir[:n]]) + epoch
                )
            torch.save(net.state_dict(), checkpoint)
        writer.close()



if __name__=="__main__":
    train()
