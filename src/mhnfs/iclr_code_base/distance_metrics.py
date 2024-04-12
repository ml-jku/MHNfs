import numpy as np
import torch


def cosineSim(
    Q,
    S,
    supportSetSize,
    scaling,
    device="cpu",
    l2Norm=True,
):
    """
    Similarity search approach, based on
    - query-, support sets split for a multi task setting
    - metric: cosine similarity
    - support-set here only consists of active molecules
    - only pytorch supported
    :param Q: query-set, torch tensor, shape[numb_tasks,*,d]
    :param S: support-set, torch tensor, shape[numb_tasks,*,d]
    :return: Predictions for each query molecule in every task
    """
    # L2 - Norm

    if l2Norm == True:
        Q_div = torch.unsqueeze(Q.pow(2).sum(dim=2).sqrt(), 2)
        Q_div[Q_div == 0] = 1  
        S_div = torch.unsqueeze(S.pow(2).sum(dim=2).sqrt(), 2)
        S_div[S_div == 0] = 1 

        Q = Q / Q_div
        S = S / S_div

    similarities = Q @ torch.transpose(S, 1, 2)

    # mask: remove padded support set artefacts
    mask = torch.zeros_like(similarities)
    for task_idx in range(S.shape[0]):
        realSize = supportSetSize[task_idx]
        if realSize > 0:
            mask[task_idx, :, :realSize] = torch.ones_like(mask[task_idx, :, :realSize])

    similarities = similarities * mask

    similaritySums = similarities.sum(
        dim=2
    )  # For every query molecule: Sum over support set molecules

    if scaling == "1/N":
        stabilizer = torch.tensor(1e-8).float()
        predictions = (
            1 / (2.0 * supportSetSize.reshape(-1, 1) + stabilizer) * similaritySums
        )
    if scaling == "1/sqrt(N)":
        stabilizer = torch.tensor(1e-8).float()
        predictions = (
            1
            / (2.0 * torch.sqrt(supportSetSize.reshape(-1, 1).float()) + stabilizer)
            * similaritySums
        )

    return predictions




distance_metrics = dict(
    {
        "cosineSim": cosineSim
    }
)
