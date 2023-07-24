import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_curve, auc, average_precision_score


def eval_metrics(probs, targets):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    assert isinstance(probs, np.ndarray)
    assert isinstance(targets, np.ndarray)

    threshold_list = []
    for i in range(1, 50):
        threshold_list.append(i / 50.0)

    if isinstance(probs, torch.Tensor) and isinstance(targets, torch.Tensor):
        fpr, tpr, thresholds = roc_curve(y_true=targets.detach().cpu().numpy(),
                                         y_score=probs.detach().cpu().numpy())
    elif isinstance(probs, np.ndarray) and isinstance(targets, np.ndarray):
        fpr, tpr, thresholds = roc_curve(y_true=targets,y_score=probs)
    else:
        print('ERROR: probs or targets type is error.')
        raise TypeError
    auc_ = auc(x=fpr, y=tpr)
    ap_ = average_precision_score(targets, probs)

    re0_5 = getROCE(probs.tolist(), targets.tolist(), 0.5)
    re1 = getROCE(probs.tolist(), targets.tolist(), 1)
    re2 = getROCE(probs.tolist(), targets.tolist(), 2)
    re5 = getROCE(probs.tolist(), targets.tolist(), 5)


    threshold_best, rec_best, pre_best,F1_best, spe_best, mcc_best, pred_bi_best = 0, 0, 0,0, 0, -1, None
    for threshold in threshold_list:
        threshold, rec, pre,F1, spe, mcc, pred_bi = th_eval_metrics_withoutAUC(threshold, probs, targets)
        if mcc > mcc_best:
            threshold_best, rec_best, pre_best,F1_best, spe_best, mcc_best, pred_bi_best = threshold, rec, pre,F1, spe, mcc, pred_bi

    return threshold_best, rec_best, pre_best,F1_best, spe_best, mcc_best, auc_, ap_, re0_5,re1,re2,re5,  pred_bi_best


def getROCE(predList,targetList,roceRate):
    p = sum(targetList)
    n = len(targetList) - p
    predList = [[index,x] for index,x in enumerate(predList)]
    predList = sorted(predList,key = lambda x:x[1],reverse = True)
    tp1 = 0
    fp1 = 0
    maxIndexs = []
    for x in predList:
        if(targetList[x[0]] == 1):
            tp1 += 1
        else:
            fp1 += 1
            if(fp1>((roceRate*n)/100)):
                break
    if p*fp1 == 0:
        roce = 0
    else:
        roce = (tp1*n)/(p*fp1)
    return roce

def th_eval_metrics_withoutAUC(threshold, probs, targets):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    assert isinstance(probs, np.ndarray)
    assert isinstance(targets, np.ndarray)

    pred_bi = np.abs(np.ceil(probs - threshold))

    try:
        tn, fp, fn, tp = confusion_matrix(targets, pred_bi).ravel()
    except:
        a = confusion_matrix(targets, pred_bi).ravel()
        a=0
    if tp >0 :
        rec = tp / (tp + fn)
    else:
        rec = 1e-8
    if tp >0:
        pre = tp / (tp + fp)
    else:
        pre = 1e-8
    if tn>0:
        spe = tn / (tn + fp)
    else:
        spe = 1e-8
    mcc = matthews_corrcoef(targets, pred_bi)
    if rec + pre > 0:
        F1 = 2 * rec * pre / (rec + pre)
    else:
        F1 = 0

    return threshold, rec, pre,F1, spe, mcc, pred_bi


def th_eval_metrics(threshold, probs, targets):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    assert isinstance(probs, np.ndarray)
    assert isinstance(targets, np.ndarray)

    fpr, tpr, thresholds = roc_curve(y_true=targets, y_score=probs)
    auc_ = auc(x=fpr, y=tpr)
    ap_ = average_precision_score(targets, probs)

    re0_5 = getROCE(probs.tolist(), targets.tolist(), 0.5)
    re1 = getROCE(probs.tolist(), targets.tolist(), 1)
    re2 = getROCE(probs.tolist(), targets.tolist(), 2)
    re5 = getROCE(probs.tolist(), targets.tolist(), 5)

    pred_bi = np.abs(np.ceil(probs - threshold))

    try:
        tn, fp, fn, tp = confusion_matrix(targets, pred_bi).ravel()
    except:
        a = confusion_matrix(targets, pred_bi).ravel()
        a = 0
    if tp > 0:
        rec = tp / (tp + fn)
    else:
        rec = 1e-8
    if tp > 0:
        pre = tp / (tp + fp)
    else:
        pre = 1e-8
    if tn > 0:
        spe = tn / (tn + fp)
    else:
        spe = 1e-8
    mcc = matthews_corrcoef(targets, pred_bi)
    if rec + pre > 0:
        F1 = 2 * rec * pre / (rec + pre)
    else:
        F1 = 0

    return threshold, rec, pre, F1, spe, mcc, auc_, ap_,re0_5,re1,re2,re5,  pred_bi


