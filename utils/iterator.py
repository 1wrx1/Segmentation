"""
Updated: 2021.06.02
New features:
(1) Introduce mixed precision training.
(2) Learning rate can be seen directly during training.
"""
import os
import random
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from utils.metrics import BinaryMetrics, MetricMeter


def train_one_epoch(model, device, train_loader, criterion, optimizer, idx, verbose=False, mixed=False):
    """
    在dataloader上完成一轮完整的迭代
    :param model: 网络模型
    :param device: cuda或cpu
    :param train_loader: 训练数据loader
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param idx: 迭代轮数
    :param verbose: 是否打印进度条
    :return: training loss
    """
    model.train()
    print('\nEpoch {} starts, please wait...'.format(idx))

    # tqdm用于显示进度条
    loader = tqdm(train_loader)
    loss_list = []
    # 用于混合精度训练
    scaler = GradScaler()

    for i, sample in enumerate(loader):
        data = sample['image'].to(device)
        mask = sample['mask'].to(device)
        if mixed:
            with autocast():
                output = model(data)
                loss = criterion(output, mask.long())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            output = model(data)
            loss = criterion(output, mask.long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        loss_list.append(loss.item())
        loader.set_postfix_str(
            'lr:{:.8f}, loss: {:.4f}'.format(optimizer.param_groups[0]['lr'], np.mean(loss_list)))

    torch.cuda.empty_cache()
    if not verbose:
        print('[ Training ] Lr:{:.8f}, Epoch Loss: {:.4f}'.format(optimizer.param_groups[0]['lr'], np.mean(loss_list)))
    return np.mean(loss_list)


# 调用torch.no_grad装饰器，验证阶段不进行梯度计算
@torch.no_grad()
def evaluate(model, device, test_loader, criterion, metric_list):
    """
    模型评估
    :param model: 网络模型
    :param device: cuda或cpu
    :param test_loader: 测试数据loader
    :param criterion: 损失函数
    :param metric_list: 评估指标列表
    :return: test loss，评估指标
    """
    model.eval()

    metric_meter = MetricMeter(metrics=metric_list)
    loss_list = []

    for i, sample in enumerate(test_loader):
        data = sample['image'].to(device)
        mask = sample['mask'].to(device)
        output = model(data)
        loss = criterion(output, mask.long())
        loss_list.append(loss.item())

        metrics = BinaryMetrics()(mask, output)
        metric_meter.update(metrics)

    print('[ Validation ] Loss: {:.4f}'.format(np.mean(loss_list)), end=' ')
    metric_meter.report(print_stats=True)
    return np.mean(loss_list), metric_meter


def set_random_seed(seed=512, benchmark=True):
    """
    设定训练随机种子
    :param seed: 随机种子
    :return: None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    if not benchmark:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
