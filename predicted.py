import torch
import torchvision
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

from VGGNet import VGG16,VGGBlock

def pre():
  # 创建一个VGG16对象
  model = VGG16((1, 32, 32), batch_norm=True)

  # 加载训练好的模型参数
  model.load_state_dict(torch.load("vgg16_mnist_model.pth"))
  # 将模型设置为评估模式
  model.eval()

  transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
  ])

  image = Image.open("img/image.png")# 加载一个手写数字图像
  # 将图像转换为张量格式
  image_tensor = transform(image).unsqueeze(1)
  # print("Image Tensor Size:", image_tensor.size())


  with torch.no_grad():
      output = model(image_tensor)

  # 获取模型输出结果的最大值和对应的类别
  max_values  = torch.max(output.data, 1)
  max_value = max_values [1]
  predicted = max_value [1]
  print("tensor:", max_values)
  print("Predicted:", int(predicted))
  
  return str(int(predicted))