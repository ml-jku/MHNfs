import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from hydra.experimental import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from argparse import ArgumentParser

from model import MHNfs


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    arguments = vars(parser.parse_args())
    return arguments


if __name__ == "__main__":
    args = get_args()
    config_name = args['config']

    GlobalHydra.instance().clear()
    initialize(config_path="../configs")
    config = compose(config_name=config_name)
    print(f'Config: {config_name}')
    print(f'Model: {config.model.name}')
    print(f'Experiment: {config.experiment_name}')
    print(f'----------- {config.comment}')

    seed_everything(config.training.seed)

    # Load data module
    dm = FSMDataModule(config)

    # Load model
    model = MHNfs(config)

    # Prepare logger
    logger = pl_loggers.WandbLogger(save_dir='./logs/', name=config.experiment_name, project=config.project_name)
    checkpoint_callback = ModelCheckpoint(monitor="dAUPRC_val", mode='max', save_top_k=1)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Setup trainer
    trainer = pl.Trainer(gpus=1, logger=logger, callbacks=[checkpoint_callback, lr_monitor],
                         max_epochs=config.training.epochs, accumulate_grad_batches=5)

    # Train
    trainer.fit(model, dm)
