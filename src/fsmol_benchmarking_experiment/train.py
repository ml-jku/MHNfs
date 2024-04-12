import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import hydra
import os
from argparse import ArgumentParser
import sys
sys.path.append('../../../MHNfs/')
from src.data.dataloader import FSMolDataModule

from src.mhnfs.models import MHNfs


@hydra.main(config_path="../mhnfs/configs", config_name="cfg")
def train(cfg):
    """
    Training loop for model training on FS-Mol.

    - A FS-Mol data-module includes dataloader for training, validation and test.
    - MHNfs defines the model which is trained
    - a pytorch lightning trainer object takes the model and the data module in performs
      model training
    - For logging, we use wandb

    inputs:
    - cfg: hydra config file
    """
    # Set seed
    seed_everything(cfg.training.seed)

    # Load data module
    dm = FSMolDataModule(cfg)

    # Load model
    model = MHNfs(cfg).to(cfg.system.ressources.device)
    model.context_embedding = model.context_embedding.to(cfg.system.ressources.device)
    model._update_context_set_embedding()

    # Prepare logger
    logger = pl_loggers.WandbLogger(
        save_dir="../../logs/", name=cfg.experiment_name, project=cfg.project_name
    )
    checkpoint_dauprc_val = ModelCheckpoint(
        monitor="dAUPRC_val", mode="max", save_top_k=1
    )
    checkpoint_dauprc_val_ma = ModelCheckpoint(
        monitor="dAUPRC_val_ma", mode="max", save_top_k=1
    )
    checkpoint_dauprc_delta = ModelCheckpoint(
        monitor="dAUPRC_train_val_delta", mode="min", save_top_k=1
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Setup trainer
    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        callbacks=[checkpoint_dauprc_val,
                   checkpoint_dauprc_val_ma,
                   checkpoint_dauprc_delta,
                   lr_monitor],
        max_epochs=cfg.training.epochs,
        accumulate_grad_batches=5,
        reload_dataloaders_every_n_epochs=1, 
    )

    # Train
    trainer.fit(model, dm)


if __name__ == "__main__":
    train()
