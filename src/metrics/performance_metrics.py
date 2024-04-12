"""


This file includes the AUC and the ΔAUC-PR performance metric to evaluate the 
performance of ML models.

The data is assumed to be provided in the form of 3 pytorch tensors:
- predictions: Tensor of shape (n_samples, 1) with the raw predictions of the model
- labels: Tensor of shape (n_samples, 1) with the true labels of the samples
- target_ids: Tensor of shape (n_samples, 1) with the target indices of the samples.
              Target indices indicate to which target the label belongs to. E.g., in a 
              multi-task setting with 10 targets, the target indices would be integers 
              between 0 and 9.

The metric functions firstly compute the the metric for each target separately and then
builds the mean over all targets. The functions return this mean value, as well as 
target-wise metric values and the target indices (from the target index the target 
names can be collected from the preprocessed data object).
"""

import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


def compute_auc_score(predictions, labels, target_ids):
    """
    Computes the AUC score for each target separately and the mean AUC score.
    """
    aucs = list()
    target_id_list = list()

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
    """
    Computes the ΔAUC-PR score for each target separately and the mean ΔAUC-PR score.
    """
    dauprcs = list()
    target_id_list = list()

    for target_idx in torch.unique(target_ids):
        rows = torch.where(target_ids == target_idx)
        preds = predictions[rows].detach()
        y = labels[rows].int()

        if torch.unique(y).shape[0] == 2:
            number_actives = y[y == 1].shape[0]
            number_inactives = y[y == 0].shape[0]
            number_total = number_actives + number_inactives

            random_clf_auprc = number_actives / number_total
            auprc = average_precision_score(
                y.numpy().flatten(), preds.numpy().flatten()
            )

            dauprc = auprc - random_clf_auprc
            dauprcs.append(dauprc)
            target_id_list.append(target_idx.item())
        else:
            dauprcs.append(np.nan)
            target_id_list.append(target_idx.item())

    return np.nanmean(dauprcs), dauprcs, target_id_list