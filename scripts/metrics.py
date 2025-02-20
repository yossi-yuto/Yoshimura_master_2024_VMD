import numpy as np
from sklearn.metrics import precision_recall_curve

def get_maxFscore_and_threshold(true_1d, pred_1d) -> tuple:
    assert len(true_1d.shape) == 1 and len(pred_1d.shape) == 1, f"No format input.shape is not 1 dimension."
    precisions, recalls, thres = precision_recall_curve(true_1d, pred_1d)
    numerator = (1 + 0.3) * (precisions * recalls)
    denom = (0.3 * precisions) + recalls
    fbetas = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
    return np.max(fbetas), thres[np.argmax(fbetas)]


def calculate_iou(ground_truth, prediction):
    """
    IoUを計算する関数
    :param ground_truth: 正解ラベル (0と1で構成されたバイナリマスク)
    :param prediction: 予測ラベル (0と1で構成されたバイナリマスク)
    :return: IoUの値
    """
    # 交差部分 (AND)
    intersection = np.logical_and(ground_truth, prediction)
    # 和集合部分 (OR)
    union = np.logical_or(ground_truth, prediction)
    # IoUの計算
    iou = np.sum(intersection) / np.sum(union)
    return iou


def calc_iou_multi_thresh(ground_truth: np.ndarray, prediction: np.ndarray) -> dict:
    """
    予測ラベルのしきい値を変化させながらIoUを計算する関数
    :param ground_truth: 正解ラベル (0と1で構成されたバイナリマスク)
    :param prediction: 予測ラベル (0と1で構成されたバイナリマスク)
    :return: IoUの値のリスト
    """
    assert ground_truth.shape == prediction.shape, f"ground_truth.shape != prediction.shape"
    assert prediction.max() <= 1 and prediction.min() >= 0, f"prediction.max() <= 1 and prediction.min() >= 0"
    iou_dict = {}
    for thres in np.arange(0.1, 1.0, 0.1):
        iou = calculate_iou(ground_truth, prediction > thres)
        iou_dict[thres] = iou
    return iou_dict