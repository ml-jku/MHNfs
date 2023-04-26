import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


def compute_auc_score(predictions, labels, target_ids):
    aucs = list()
    target_id_list = list()

    target_idx: torch.Tensor
    for target_idx in torch.unique(target_ids):
        rows = torch.where(target_ids == target_idx)
        preds = predictions[rows].detach()
        y = labels[rows]

        if torch.unique(y).shape[0] == 2:
            auc = roc_auc_score(y, preds)
            aucs.append(auc)
            target_id_list.append(target_idx.item())
        else:
            aucs.append(np.nan)
            target_id_list.append(target_idx.item())
    return np.nanmean(aucs), aucs, target_id_list


def compute_dauprc_score(predictions, labels, target_ids):
    dauprcs = list()
    target_id_list = list()

    target_ids: torch.Tensor
    for target_idx in torch.unique(target_ids):
        rows = torch.where(target_ids == target_idx)
        preds = predictions[rows].detach()
        y = labels[rows].int()

        if torch.unique(y).shape[0] == 2:
            number_actives = y[y == 1].shape[0]
            number_inactives = y[y == 0].shape[0]
            number_total = number_actives + number_inactives

            random_clf_auprc = number_actives / number_total
            auprc = average_precision_score(y.numpy().flatten(), preds.numpy().flatten())

            dauprc = auprc - random_clf_auprc
            dauprcs.append(dauprc)
            target_id_list.append(target_idx.item())
        else:
            dauprcs.append(np.nan)
            target_id_list.append(target_idx.item())

    return np.nanmean(dauprcs), dauprcs, target_id_list
