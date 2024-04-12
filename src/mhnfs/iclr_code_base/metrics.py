import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

def cosine_sim(u, v):

    u = u / np.linalg.norm(u, axis=1).reshape(-1, 1)
    v = v / np.linalg.norm(v, axis=1).reshape(-1, 1)

    y_pred = u @ v.T

    return y_pred

def deltaAUPRC_score_singleTask(predictions, labels, numberActives_perTask, numberInactives_perTask):
    """
    Compute delta AUPRC score. AUPRC of a model is compared to the performance of a random classifier
    :param predictions: model predictions
    :param labels: labels of query molecules
    :param numberActives_perTask: int
    :param numberInactives_perTask: int
    :return:
    """
    random_clf_auprc = numberActives_perTask / (numberActives_perTask + numberInactives_perTask)
    auprc = average_precision_score(labels, predictions)

    deltaAuprc = auprc - random_clf_auprc
    return deltaAuprc

def deltaAUPRC_score(predictions, labels, querySet_actives_taskList, querySet_inactives_taskList,
                     targetNames_list):
    """
    Compute delta AUPRC score. AUPRC of a model is compared to the performance of a random classifier
    - Multi-task setting
    :param predictions: model predictions
    :param labels: labels of query molecules
    :param querySet_actives_taskList: int
    :param querySet_inactives_taskList: int
    :return:
    """
    deltaAUPRCs = list()
    targetNames = list()

    number_tasks = predictions.shape[0]
    for task_idx in range(number_tasks):
        nbrActives = querySet_actives_taskList[task_idx]
        nbrInactives = querySet_inactives_taskList[task_idx]
        nbrTotal = nbrActives + nbrInactives

        preds = predictions[task_idx][:nbrTotal]
        y = labels[task_idx][:nbrTotal]

        healthy_datapoint_idx = np.where(preds.numpy().astype('str') != 'nan')
        preds = preds[healthy_datapoint_idx[0]]
        y = y[healthy_datapoint_idx[0]].int()

        random_clf_auprc = nbrActives / nbrTotal
        auprc = average_precision_score(y.numpy().flatten(), preds.numpy().flatten())

        deltaAuprc = auprc - random_clf_auprc
        deltaAUPRCs.append(deltaAuprc)

        targetNames.append(targetNames_list[task_idx])

    return np.mean(deltaAUPRCs), deltaAUPRCs, targetNames

def auc_score(predictions, labels, querySet_actives_taskList, querySet_inactives_taskList,
                     targetNames_list):
    """
        Compute AUC score
        - Multi-task setting
        - No missing values in dataset -> No masking necessary
        :param predictions: model predictions
        :param labels: labels of query molecules
        :param querySet_actives_taskList: int
        :param querySet_inactives_taskList: int
        :param targetNames_list: List of target_names
        :return: tuple: target-wise mean over AUCs, list with auc scores (target-wise), target names
        """
    AUCs = list()

    number_tasks = predictions.shape[0]
    for task_idx in range(number_tasks):
        nbrActives = querySet_actives_taskList[task_idx]
        nbrInactives = querySet_inactives_taskList[task_idx]
        nbrTotal = nbrActives + nbrInactives

        preds = predictions[task_idx][:nbrTotal]
        y = labels[task_idx][:nbrTotal]

        auc = roc_auc_score(y, preds)

        AUCs.append(auc)

    return np.mean(AUCs), AUCs, targetNames_list

def auc_score_train(predictions, labels, target_ids):
    AUCs = list()
    target_id_list = list()

    for target_idx in torch.unique(target_ids):
        rows = torch.where(target_ids == target_idx)
        preds = predictions[rows].detach()
        y = labels[rows]

        if torch.unique(y).shape[0] == 2:
            auc = roc_auc_score(y,preds)
            AUCs.append(auc)
            target_id_list.append(target_idx.item())
        else:
            AUCs.append(np.nan)
            target_id_list.append(target_idx.item())
    return np.nanmean(AUCs), AUCs, target_id_list

def deltaAUPRC_score_train(predictions, labels, target_ids):
    deltaAUPRCs = list()
    target_id_list = list()

    for target_idx in torch.unique(target_ids):
        rows = torch.where(target_ids == target_idx)
        preds = predictions[rows].detach()
        y = labels[rows].int()

        if torch.unique(y).shape[0] == 2:
            nbrActives = y[y == 1].shape[0]
            nbrInactives = y[y == 0].shape[0]
            nbrTotal = nbrActives + nbrInactives

            random_clf_auprc = nbrActives / nbrTotal
            auprc = average_precision_score(y.numpy().flatten(), preds.numpy().flatten())

            deltaAuprc = auprc - random_clf_auprc

            deltaAUPRCs.append(deltaAuprc)
            target_id_list.append(target_idx.item())
        else:
            deltaAUPRCs.append(np.nan)
            target_id_list.append(target_idx.item())


    return np.nanmean(deltaAUPRCs), deltaAUPRCs, target_id_list




