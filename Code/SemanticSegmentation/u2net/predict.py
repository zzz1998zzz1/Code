import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms

from src import u2net_full


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # model weights save path
    weights_path = "save_weights/model_best.pth"
    # path of image to be predict
    img_path = ".."

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
        plt.imshow(seg_img_inv)
        plt.show()

        cv2.imwrite("pred_blackbackground.png", cv2.cvtColor(seg_img.astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite("pred_result.png", cv2.cvtColor(seg_img_inv.astype(np.uint8), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
