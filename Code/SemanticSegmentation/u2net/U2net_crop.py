# -- coding: utf-8 --
import os
import time
from shutil import rmtree

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from src import u2net_full

# model weights save path
weight_path = "save_weights/model_best.pth"
# path of image to be segmented
img_path = ".."
# single seed image save path
save_root = ".."


def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def seg(img_path, save_path):
    weights_path = weight_path
    img_path = img_path
    threshold = 0.5

    assert os.path.exists(img_path), f"image file {img_path} dose not exists."

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(320),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    img_gt = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    origin_img = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)

    h, w = origin_img.shape[:2]
    img = data_transform(origin_img)
    img = torch.unsqueeze(img, 0).to(device)  # [C, H, W] -> [1, C, H, W]

    model = u2net_full()
    weights = torch.load(weights_path, map_location='cpu')
    if "model" in weights:
        model.load_state_dict(weights["model"])
    else:
        model.load_state_dict(weights)
    model.to(device)
    model.eval()

    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        pred = model(img)
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))
        pred = torch.squeeze(pred).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]

        pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        pred_mask = np.where(pred > threshold, 1, 0)
        origin_img = np.array(origin_img, dtype=np.uint8)
        seg_img = origin_img * pred_mask[..., None]
        seg_img_inv = np.where(seg_img == 0, 255, seg_img)

        seg_img_gray = cv2.cvtColor(np.uint8(seg_img), cv2.COLOR_BGR2GRAY)
        contours, hre = cv2.findContours(seg_img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print(len(contours))
        result = np.copy(seg_img_inv)
        contour_image = cv2.drawContours(result, contours, -1, (255, 0, 0), 20)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            roi = seg_img_inv[y:y + h, x:x + w]

            filename = "1.png"
            while os.path.exists(os.path.join(save_path, filename)):
                filename = str(int(filename.split(".", 2)[0]) + 1) + ".png"
            print(os.path.join(save_path, filename))

            cv2.imencode('.png', cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_RGB2BGR))[1].tofile(
                os.path.join(save_path, filename))

            cv2.rectangle(result, (x, y), (x + w, y + h), (65, 105, 255), 20)


def main():
    mk_file(save_root)
    for root, dirs, files in os.walk(img_path):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            save_dir = os.path.join(save_root, dir)
            mk_file(save_dir)
            for img in os.listdir(dir_path):
                seg(os.path.join(dir_path, img), save_dir)


if __name__ == '__main__':
    main()
