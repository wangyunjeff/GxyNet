import argparse
import glob
import os

import cv2
import numpy as np
import PIL
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn

from datasets import KeyPointDatasets
from model import KeyPointModel
from coord import transfer_target

def pool_nms(heat, kernel=7):
    pad = (kernel - 1) // 2
    # pad = 1
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description="model path")
parser.add_argument('--model', type=str, default="./logs/epoch_99_0.054.pt")

args = parser.parse_args()


SIZE = 256, 256

transforms_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4372, 0.4372, 0.4373],
                         std=[0.2479, 0.2475, 0.2485])
])

datasets_test = KeyPointDatasets(root_dir="./data", transforms=transforms_test)

dataloader_test = DataLoader(
    datasets_test, batch_size=4, shuffle=True, collate_fn=datasets_test.collect_fn)

model = KeyPointModel()
model.eval()

model.load_state_dict(torch.load(args.model))

img_list = glob.glob(os.path.join("./data/detection_images", "*.bmp"))

save_path = "./output"

img_tensor_list = []
img_name_list = []

for i in range(len(img_list)):
    img_path = img_list[i]
    img_name = os.path.basename(img_path)
    img_name_list.append(img_name)

    img = cv2.imread(img_path)
    img_tensor = transforms_test(img)
    img_tensor_list.append(img_tensor)

img_tensor_list = torch.stack(img_tensor_list, 0)

print(img_tensor_list.shape)

# part of it
img_tensor_list = img_tensor_list
img_name_list = img_name_list
# img_tensor_list = img_tensor_list[0:50]
# img_name_list = img_name_list[0:50]

if torch.cuda.is_available():
    model = model.cuda()
    img_tensor_list = img_tensor_list.cuda()

heatmap = model(img_tensor_list)

import matplotlib.pyplot as plt
a = heatmap[1][0].cpu().detach().numpy()
plt.imshow(a)
plt.show()
a_maxpool = pool_nms(heatmap)
a_maxpool = a_maxpool[1][0].cpu().detach().numpy()
plt.imshow(a_maxpool)
plt.show()




heatmap = torch.sigmoid(heatmap)
heatmap = heatmap.squeeze().cpu()

pred = heatmap.unsqueeze(3).detach().numpy()

landmark_coord = transfer_target(pred, thresh=0.1, n_points=1)

print(landmark_coord.shape)

bs = img_tensor_list.shape[0]

# for i in range(bs):
#     print(landmark_coord[i])

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


for i in range(bs):
    img_path = img_list[i]
    img = cv2.imread(img_path)

    img = cv2.resize(img, (256, 256))

    single_map = heatmap[i]

    hm = single_map.detach().numpy()

    hm = np.maximum(hm, 0)
    hm = hm/np.max(hm)
    hm = normalization(hm)

    hm = np.uint8(255 * hm)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm = cv2.resize(hm, (256, 256))

    superimposed_img = hm * 0.2 + img

    coord_x, coord_y = landmark_coord[i]

    cv2.circle(superimposed_img, (int(coord_x), int(coord_y)), 2, (0, 0, 0), thickness=-1)

    # print("./output/%s_out.jpg" % (img_name_list[i]))

    cv2.imwrite("./output/%s_out.jpg" % (img_name_list[i]), superimposed_img)

print("done")