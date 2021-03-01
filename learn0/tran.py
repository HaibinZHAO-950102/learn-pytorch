from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs')

img_path = '/Users/haibinzhao/Desktop/TECO/Software/Pytorch/learn0/train/cat.0.jpg'
img = Image.open(img_path)
print(img)

totensor = transforms.ToTensor()
img_tensor = totensor(img)
writer.add_image('ToTensor', img_tensor)

print(img_tensor[0][0][0])
normen = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = normen(img_tensor)
print(img_norm[0][0][0])
writer.add_image('Normalize', img_norm)

print(img.size)
resize_img = transforms.Resize((300,300))
img_resize = resize_img(img_norm)
writer.add_image('Resize', img_resize)

trans_compose = transforms.Compose([totensor, normen, resize_img])
img_compose = trans_compose(img)
writer.add_image('Compose', img_compose)

rancrop = transforms.RandomCrop(50)
for i in range(10):
    img_crop = rancrop(img_tensor)
    writer.add_image("randomcrop", img_crop, i)

writer.close()