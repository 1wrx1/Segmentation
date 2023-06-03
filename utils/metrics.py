import numpy as np
import torch
import torch.nn as nn
from utils.surface_distance.metrics import compute_surface_distances, compute_robust_hausdorff, compute_average_surface_distance


class BinaryMetrics():
    """
    Compute common metrics for binary segmentation, including overlap metrics, distance metrics and MAE
    NOTE: batch size must be set to one for accurate measurement, batch size larger than one may cause errors!
    """
    def __init__(self, eps=1e-5, resolution=(1, 1), inf_result=np.nan):
        self.eps = eps
        self.resolution = resolution
        self.inf_result = inf_result

    def _check_inf(self, result):
        if result == np.inf:
            return self.inf_result
        else:
            return result

    def _calculate_overlap_metrics(self, gt, pred):
        output = pred.view(-1, )
        target = gt.view(-1, ).float()

        tp = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1 - target))  # FP
        fn = torch.sum((1 - output) * target)  # FN
        tn = torch.sum((1 - output) * (1 - target))  # TN

        pixel_acc = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)
        dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        precision = (tp + self.eps) / (tp + fp + self.eps)
        recall = (tp + self.eps) / (tp + fn + self.eps)
        specificity = (tn + self.eps) / (tn + fp + self.eps)

        metric_dict = dict()
        metric_dict['pixel_acc'] = pixel_acc.item()
        metric_dict['dice'] = dice.item()
        metric_dict['precision'] = precision.item()
        metric_dict['recall'] = recall.item()
        metric_dict['specificity'] = specificity.item()

        return metric_dict

    def _calculate_distance_metrics(self, gt, pred):
        # shape: (N, C, H, W)
        gt_class = gt[0, ...].cpu().numpy().astype(np.int).astype(np.bool)  # (H, W)
        pred_class = pred[0, 0, ...].cpu().numpy().astype(np.int).astype(np.bool)  # (H, W)
        surface_distance_dict = compute_surface_distances(gt_class, pred_class, self.resolution)
        distances = surface_distance_dict['distances_pred_to_gt']
        mean_surface_distance = self._check_inf(np.mean(distances))

        # compute Hausdorff distance (95 percentile)
        hd95 = self._check_inf(compute_robust_hausdorff(surface_distance_dict, percent=95))
        ASD=self._check_inf(compute_average_surface_distance(surface_distance_dict))

        metric_dict = dict()
        metric_dict['mean_surface_distance'] = mean_surface_distance
        metric_dict['hd95'] = hd95
        metric_dict['ASD'] = ASD

        return metric_dict

    def _calculate_mae(self, gt, pred):
        # shape: (N, C, H, W)
        residual = torch.abs(gt.unsqueeze(1) - pred)
        mae = torch.mean(residual, dim=(2, 3)).squeeze().detach().cpu().numpy()

        metric_dict = dict()
        metric_dict['mae'] = mae
        return metric_dict

    def __call__(self, y_true, y_pred):
        # y_true: (N, H, W)
        # y_pred: (N, 1, H, W)
        sigmoid_pred = nn.Sigmoid()(y_pred)
        class_pred = (sigmoid_pred > 0.5).float().to(y_pred.device)

        assert class_pred.shape[1] == 1, 'Predictions must contain only one channel' \
                                             ' when performing binary segmentation'

        overlap_metrics = self._calculate_overlap_metrics(y_true.to(y_pred.device, dtype=torch.float), class_pred)
        distance_metrics = self._calculate_distance_metrics(y_true, class_pred)
        mae = self._calculate_mae(y_true, sigmoid_pred)

        metrics = {**overlap_metrics, **distance_metrics, **mae}

        return metrics


class MetricMeter(object):
    """
    Metric记录器
    """
    def __init__(self, metrics):
        self.metrics = metrics
        self.initialization()

    def initialization(self):
        for metric in self.metrics:
            exec('self.' + metric + '=[]')

    def update(self, metric_dict):
        """
        将新的metric字典传入，更新记录器
        :param metric_dict: 指标字典
        :return: None
        """
        for metric_key, metric_value in metric_dict.items():
            try:
                exec('self.' + metric_key + '.append(metric_value)')
            except:
                continue

    def report(self, print_stats=True):
        """
        汇报目前记录的指标信息
        :param print_stats: 是否将指标信息打印在终端
        :return: report_str
        """
        report_str = ''
        for metric in self.metrics:
            metric_mean = np.nanmean(eval('self.' + metric), axis=0)
            metric_std = np.nanstd(eval('self.' + metric), axis=0)
            if print_stats:
                stats = metric + ': {} ± {};'.format(np.around(metric_mean, decimals=4),
                                                     np.around(metric_std, decimals=4))
                print(stats, end=' ')
                report_str += stats
        return report_str
