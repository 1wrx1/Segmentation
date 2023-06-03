"""
Updated: 2021.06.02
New features:
(1) Necessary code modification, now visualization only runs on the validation set.
"""
import warnings
warnings.filterwarnings('ignore')
import os
import re
import yaml
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from skimage.io import imsave
from skimage.transform import resize
from models.u_net import UNet, AttentionUNet, NestedUNet
from utils.data_pipeline import SegmentationDataset, ToTensor, Rescale
from torchvision.transforms import Compose


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', type=str, default='data/images_2d',
                        help='path for 2d image files')
    parser.add_argument('-m', '--mask_dir', type=str, default='data/masks_2d',
                        help='path for 2d mask files')
    parser.add_argument('-c', '--checkpoint', type=str, default='checkpoints/checkpoint_2.pth',
                        help='checkpoint path')
    parser.add_argument('-s', '--size', type=int, default=448, help='input image size')
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint
    image_root = args.image_dir
    mask_root = args.mask_dir
    size = (args.size, args.size)

    output_dir = './visualization'
    os.makedirs(output_dir, exist_ok=True)

    with open('./config.cfg', 'r') as f:
        cfg = yaml.safe_load(f)
        print("successfully loaded config file: ", cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = {'unet': UNet, 'att_unet': AttentionUNet, 'unet++': NestedUNet}
    model_name = cfg['MODEL']['TYPE']  # 模型类型
    backbone = cfg['MODEL']['BACKBONE']  # 网络骨架
    dataseed = cfg['TRAIN']['DATASEED']  # 数据划分种子

    model = model_dict[model_name](num_classes=1, backbone=backbone, pretrained=False).to(device).eval()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

    subject_list = [re.findall('(\d+)', file)[0] for file in os.listdir('./data/images')]
    train_list, val_list = train_test_split(subject_list, test_size=0.2, random_state=dataseed)

    transform = Compose([Rescale(size), ToTensor()])
    valset = SegmentationDataset(image_root, mask_root, val_list, transform=None, vis=True)

    with torch.no_grad():
        for index, (sample, data_id) in enumerate(valset):
            img = sample['image']
            msk = (sample['mask'] * 255).astype(np.uint8)
            original_shape = img.shape[:2]
            transformed_sample = transform({'image': img, 'mask': msk})

            input_data = transformed_sample['image'].unsqueeze(0).to(device)
            # 执行前馈与sigmoid激活获得概率图
            output = torch.sigmoid(model(input_data))
            # 将概率图转为channel_last格式与自然图像保持一致
            pred = output.permute(0, 2, 3, 1).squeeze().cpu().numpy()
            pred_rescaled = resize(pred, original_shape)
            # plt.figure(figsize=(12, 4))
            # plt.subplot(1,3,1)
            # plt.imshow(image, cmap='gray')
            # plt.subplot(1,3,2)
            # plt.imshow(mask, cmap='gray')
            # plt.subplot(1,3,3)
            # plt.imshow(pred, cmap='gray')
            # plt.show()
            imsave(os.path.join(output_dir, '{}-image.png'.format(data_id)), img)
            imsave(os.path.join(output_dir, '{}-mask.png'.format(data_id)), msk)
            imsave(os.path.join(output_dir, '{}-{}-pred.png'.format(data_id, model_name)),
                   (255 * pred_rescaled).astype(np.uint8))

            if (index + 1) % 50 == 0:
                print('Processed {} / {}'.format(index + 1, len(valset)))


if __name__ == '__main__':
    main()
