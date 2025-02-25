import numpy as np
from sklearn.metrics import precision_recall_curve

def get_maxFscore_and_threshold(true_1d, pred_1d) -> tuple:
    assert len(true_1d.shape) == 1 and len(pred_1d.shape) == 1, f"No format input.shape is not 1 dimension."
    precisions, recalls, thres = precision_recall_curve(true_1d, pred_1d)
    numerator = (1 + 0.3) * (precisions * recalls)
    denom = (0.3 * precisions) + recalls
    fbetas = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
    return np.max(fbetas), thres[np.argmax(fbetas)]

def get_maxFscore_and_threshold_multi_beta(true_1d: np.ndarray, pred_1d: np.ndarray, betas: list=[0.3, 0.5, 1.0]) -> dict:
    """
    各 beta 値に対して、最大の F スコアと対応する閾値を計算して辞書で返す関数

    :param true_1d: np.ndarray, shape=(n,)
    :param pred_1d: np.ndarray, shape=(n,)
    :param beta: list of float, F スコアの計算で使用する beta 値のリスト
    :return: dict, 各 beta をキーとして、{"max_fscore": 最大の F スコア, "threshold": そのときの閾値} を値に持つ辞書
    """
    precisions, recalls, thres = precision_recall_curve(true_1d, pred_1d)
    
    # precision と recall は thres より 1 つ多いので、閾値と対応付けるために最後の要素を除外
    precisions_cut = precisions[:-1]
    recalls_cut = recalls[:-1]
    
    result :dict = {}
    for _beta in betas:
        numerator = (1 + _beta) * (precisions_cut * recalls_cut)
        denom = (_beta * precisions_cut) + recalls_cut
        fbetas = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
        max_index = np.argmax(fbetas)
        max_fscore = fbetas[max_index]
        corresponding_threshold = thres[max_index]
        result[_beta] = {"max_fscore": max_fscore, "threshold": corresponding_threshold}
    
    return result


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


def calc_iou_multi_thresh(ground_truth: np.ndarray, prediction: np.ndarray, thresholds :list = [0.3, 0.5, 0.7]) -> dict:
    """
    予測ラベルのしきい値を変化させながらIoUを計算する関数
    :param ground_truth: 正解ラベル (0と1で構成されたバイナリマスク)
    :param prediction: 予測ラベル (0と1で構成されたバイナリマスク)
    :return: IoUの値のリスト
    """
    assert ground_truth.shape == prediction.shape, f"ground_truth.shape != prediction.shape"
    assert prediction.max() <= 1 and prediction.min() >= 0, f"prediction.max() <= 1 and prediction.min() >= 0"
    iou_dict = {}
    for thres in thresholds:
        iou = calculate_iou(ground_truth, prediction > thres)
        iou_dict[thres] = iou
    return iou_dict