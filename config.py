import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
object_path = './demo.png'
backgrund_folder='./data/plane'
target_folder ="./data/object_detection_segment/object_detection/"
checkpoint ="./data/chapter_two/net.pth"
scale =[0.25,0.4]
num =[1,2,3]
img_size =300
batch_size =64
num_epoch =50
lr = 0.001
epoch_Ir = [(30,0.01),(30,0.001),(50,0.0001)]