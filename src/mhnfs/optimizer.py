"""
This file includes optimizer and learning rate schedulers used for model training.
"""

import torch


def define_opimizer(config, parameters):
    if config.model.training.optimizer == "AdamW":
        base_optimizer = torch.optim.AdamW
    elif config.model.training.optimizer == "SGD":
        base_optimizer = torch.optim.SGD
    else:
        base_optimizer = torch.optim.Adam

    optimizer = base_optimizer(
        parameters,
        lr=config.model.training.lr,
        weight_decay=config.model.training.weightDecay,
    )

    if config.model.training.lrScheduler.usage:
        lrs_1 = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=config.model.training.lr, total_iters=5
        )
        lrs_2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.994)

        lrs = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[lrs_1, lrs_2], milestones=[40]
        )

        lr_dict = {"scheduler": lrs, "monitor": "loss_val"}

        return [optimizer], [lr_dict]
    else:
        return optimizer
