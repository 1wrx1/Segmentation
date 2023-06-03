"""
Updated: 2021.06.10
New features:
(1) Introduce breakpoint training continuation function.
(2) Introduce automatic mixed precision training to save video memory. It should be noted that mixed precision works
the best with Turing/Ampere GeForce GPUs with Tensor-Cores. For GPUs without Tensor-Cores, only modest speed-up is
observed. For more details about mixed precision training please refer to the docs.
"""
import os
import re
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from sklearn.model_selection import train_test_split
from models.binary_loss import *
from models.u_net import UNet, AttentionUNet, NestedUNet, DLinkNet
from utils.data_pipeline import *
from utils.iterator import train_one_epoch, evaluate, set_random_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs for training')
    parser.add_argument('--interval', type=int, default=1,
                        help='interval for validation')
    parser.add_argument('-s', '--size', type=int, default=480, help='input image size')
    parser.add_argument('--mixed', action="store_true",
                        help='whether to use mixed precision training to save video memory')
    parser.add_argument('-c', '--checkpoint', type=str, help='model checkpoint used for breakpoint continuation')
    parser.add_argument('--ncpu', type=int, default=4,
                        help='number of workers for dataloader, larger num workers requests more memory')
    parser.add_argument('--benchmark', action="store_true",default=True,
                        help='whether to use cudnn benchmark to speed up convolution operations')
    return parser.parse_args()


def main():


    # 模型存储路径
    model_savedir = './checkpoints'
    if not os.path.exists(model_savedir):
        os.makedirs(model_savedir)

    # 解析cfg文件读入训练超参数
    with open('./config.cfg', 'r') as f:
        cfg = yaml.safe_load(f)
        print("successfully loaded config file: ", cfg)

    seed = cfg['TRAIN']['SEED']  # 随机种子
    dataseed = cfg['TRAIN']['DATASEED']  # 数据划分随机种子，用于蒙特卡洛交叉验证

    batch_size = cfg['TRAIN']['BATCHSIZE']  # batch size
    lr = cfg['TRAIN']['LR']  # 基准学习率
    decay = cfg['TRAIN']['DECAY']  # l2正则化

    data_address = cfg['MODEL']['DATA']  # Data Set
    model_name = cfg['MODEL']['TYPE']  # 模型类型
    backbone = cfg['MODEL']['BACKBONE']  # 网络骨架
    is_pretrained = cfg['MODEL']['PRETRAINED']  # 是否预训练
    loss_function = cfg['MODEL']['LOSSFUNCTION']  # Loss Function

    # 数据存储路径
    train_image_root = r'D:\code\Segmentation_Code_new\data\train' + '/images'
    train_mask_root = r'D:\code\Segmentation_Code_new\data\train' + '/masks'
    test_image_root = r'D:\code\Segmentation_Code_new\data\test' + '/images'
    test_mask_root = r'D:\code\Segmentation_Code_new\data\test' + '/masks'

    # 解析argparse用于修改训练中的参数
    args = parse_args()
    num_epochs = args.num_epochs  # 训练迭代轮数
    interval = args.interval  # 验证间隔
    image_size = args.size  # 输入图像分辨率，更高的分辨率一般会取得更佳的效率，但是会带来更多的显存占用
    is_mixed = args.mixed  # 是否使用混合精度训练，可以节省显存
    benchmark = args.benchmark  # 是否使用cudnn benchmark加速卷积算法
    # 使用cudnn benchmark会加速训练过程，代价是引入一定随机性，一定程度上影响训练的复现
    # 在使用混合精度训练时，如果未开启benchmark，deterministic convolutions会非常慢，建议开启benchmark加速
    num_workers = args.ncpu  # python多线程数据加载的线程数量
    start_epoch = -1  # 默认的开始轮数为0
    print('Training Settings: Epochs-{}, Interval-{}, Image Size-{}'.format(
        num_epochs, interval, image_size, is_mixed
    ))
    print('Mixed precision status: Mixed Precision-{}, CUDNN Benchmark Acceleration-{}'.format(is_mixed, benchmark))

    # 使用的评估指标列表
    metric_list = ['pixel_acc', 'dice', 'precision', 'recall', 'specificity', 'mean_surface_distance', 'hd95','ASD']
    model_dict = {'unet': UNet, 'att_unet': AttentionUNet, 'unet++': NestedUNet, 'D_LinkNet': DLinkNet}
    loss_function_dict={'TverskyLoss':TverskyLoss, 'FocalTverskyLoss':FocalTverskyLoss, 'DiceLoss':DiceLoss,
                        'Focal_DiceLoss':Focal_DiceLoss, 'WBCE_DiceLoss':WBCE_DiceLoss,
                        'WBCEWithLogitLoss':WBCEWithLogitLoss, 'FocalLoss':FocalLoss}
    data_address_dic=['Brainstem','Eye_L','Eye_R','Lens_L','Lens_R','Mandible_L','Mandible_R',
                   'OpticChiasm','OpticNerve_L','OpticNerve_R','Parotid_L','Parotid_R','Pituitary',
                   'SpinalCord','TempLobe_L','TempLobe_R','Thyroid','TMJ_L','TMJ_R','Trachea']
    # 设置随机种子保证结果的可复现性
    set_random_seed(seed=seed, benchmark=benchmark)

    # 程序运行在cuda/cpu上
    # nvidia-smi to decide which cuda to use
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # Choose the cuda you use
    print('using device {}'.format(device))

    # 训练/验证数据预处理
    # TODO: 自行设定数据增强组合，提升分割性能
    train_transform = Compose([Rescale((512, 512)), RandomCrop((image_size, image_size)),ToTensor()])
    val_transform = Compose([Rescale((image_size, image_size)), ToTensor()])

    # 划分训练集与验证集
    train_list = [re.findall('(\d+)', file)[0] for file in os.listdir(train_image_root)]
    val_list = [re.findall('(\d+)', file)[0] for file in os.listdir(test_image_root)]

    # 定义dataset与dataloader
    trainset = SegmentationDataset(train_image_root, train_mask_root, train_list, transform=train_transform)
    valset = SegmentationDataset(test_image_root, test_mask_root, val_list, transform=val_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=num_workers)

    # 定义模型
    # TODO：自行选择合适的模型结构与backbone提升模型性能
    model = model_dict[model_name](num_classes=1, backbone=backbone, pretrained=is_pretrained).to(device)
    print('Training {} with {} backbone'.format(model_name, backbone))

    # 设置基准学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    # 损失函数与学习率调节器
    # TODO：自行选择合适的损失函数与学习率调节器提升分割性能
    criterion = loss_function_dict[loss_function]()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[num_epochs // 3, 2 * num_epochs // 3],
                                                     gamma=0.1,
                                                     last_epoch=start_epoch)

    # 如果执行断点继续训练，则解析checkpoint并恢复训练状态
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)

        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']

    # 进行训练与验证
    for epoch in range(start_epoch + 1, num_epochs):
        # 执行训练，每轮训练后更新学习率
        train_one_epoch(model, device, train_loader, criterion, optimizer, epoch + 1, True, is_mixed)
        scheduler.step()
        # 如果当前轮数需要验证，则执行验证并保存checkpoint
        if ((epoch + 1) % interval == 0) or (epoch==0):
            # 执行验证
            evaluate(model, device, val_loader, criterion, metric_list)
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": epoch
            }
            # 保存训练checkpoint
            torch.save(checkpoint, os.path.join(model_savedir, '{}_checkpoint_{}.pth'.format(model_name, epoch + 1)))
    # 最终checkpoint
    final_checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        "epoch": num_epochs - 1
    }
    torch.save(final_checkpoint, os.path.join(model_savedir, '{}_checkpoint_final.pth'.format(model_name)))


if __name__ == '__main__':
    main()
