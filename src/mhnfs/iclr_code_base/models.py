import pytorch_lightning as pl
from omegaconf import OmegaConf

import numpy as np
from scipy import sparse
import torch

import sys
sys.path.append(".")

from src.mhnfs.iclr_code_base.losses import losses

from src.mhnfs.iclr_code_base.modules import (
    EncoderBlock,
    LayerNormalizingBlock,
    IterRefEmbedding,
    TransformerEmbedding,
)
from src.mhnfs.iclr_code_base.modules import (
    HopfieldBlock_chemTrainSpace,
    FullTransformerPredRetrieval,
)
from src.mhnfs.iclr_code_base.metrics import (
    auc_score_train,
    deltaAUPRC_score_train,
)
from src.mhnfs.iclr_code_base.distance_metrics import distance_metrics
from src.mhnfs.optimizer import define_opimizer

from hydra.experimental import compose, initialize
from hydra.core.global_hydra import GlobalHydra

import wandb


class ClassicSimSearch(pl.LightningModule):
    def __init__(self, config: OmegaConf):
        super(ClassicSimSearch, self).__init__()

        # Config
        self.config = config

        # Similarity Block
        self.similarity_function = distance_metrics[config.model.similarityBlock.type]

        # Output function
        self.sigmoid = torch.nn.Sigmoid()

    def forward(
        self,
        queryMols,
        supportMolsActive,
        supportMolsInactive,
        supportSetActivesSize=0,
        supportSetInactivesSize=0,
    ):
        # Similarities:
        predictions_supportActives = self.similarity_function(
            queryMols,
            supportMolsActive,
            supportSetActivesSize,
            device=self.device,
            scaling=self.config.model.similarityBlock.scaling,
            l2Norm=self.config.model.similarityBlock.l2Norm,
        )
        # if (self.config.targetSet.numberInactives != 0 or self.config.targetSet.ratioInactives != -1.):
        _predictions_supportInactives = self.similarity_function(
            queryMols,
            supportMolsInactive,
            supportSetInactivesSize,
            device=self.device,
            scaling=self.config.model.similarityBlock.scaling,
            l2Norm=self.config.model.similarityBlock.l2Norm,
        )
        predictions = predictions_supportActives - _predictions_supportInactives

        return predictions


class NeuralSearch(pl.LightningModule):
    def __init__(self, config: OmegaConf):
        super(NeuralSearch, self).__init__()

        # Config
        self.config = config

        # Loss functions
        self.LossFunction = losses[config.model.training.loss]

        # Hyperparameter
        self.save_hyperparameters(config)

        # Encoder
        self.encoder = EncoderBlock(config)

        # Layernormalizing-block
        if self.config.model.layerNormBlock.usage == True:
            self.layerNormBlock = LayerNormalizingBlock(config)

        # Similarity Block
        self.similarity_function = distance_metrics[config.model.similarityBlock.type]

        # Output function
        self.sigmoid = torch.nn.Sigmoid()
        self.prediction_scaling = config.model.prediction_scaling

    def forward(
        self,
        queryMols,
        supportMolsActive,
        supportMolsInactive,
        supportSetActivesSize=0,
        supportSetInactivesSize=0,
    ):
        # Embeddings
        query_embedding = self.encoder(queryMols)
        supportActives_embedding = self.encoder(supportMolsActive)
        supportInactives_embedding = self.encoder(
            supportMolsInactive
        )  # Todo: add if clause below

        # if (self.config.targetSet.numberInactives != 0 or self.config.targetSet.ratioInactives != -1.):
        #    supportInactives_embedding = self.encoder(supportMolsInactive)

        # Layer normalization:
        if self.config.model.layerNormBlock.usage == True:
            # if (self.config.targetSet.numberInactives != 0 or self.config.targetSet.ratioInactives != -1.):
            (
                query_embedding,
                supportActives_embedding,
                supportInactives_embedding,
            ) = self.layerNormBlock(
                query_embedding, supportActives_embedding, supportInactives_embedding
            )
            # else:
            #    (query_embedding, supportActives_embedding,
            #     supportInactives_embedding) = self.layerNormBlock(query_embedding, supportActives_embedding,
            #                                                       None)
        # Similarities:
        predictions_supportActives = self.similarity_function(
            query_embedding,
            supportActives_embedding,
            supportSetActivesSize,
            device=self.device,
            scaling=self.config.model.similarityBlock.scaling,
            l2Norm=self.config.model.similarityBlock.l2Norm,
        )
        # if (self.config.targetSet.numberInactives != 0 or self.config.targetSet.ratioInactives != -1.):
        _predictions_supportInactives = self.similarity_function(
            query_embedding,
            supportInactives_embedding,
            supportSetInactivesSize,
            device=self.device,
            scaling=self.config.model.similarityBlock.scaling,
            l2Norm=self.config.model.similarityBlock.l2Norm,
        )
        # predictions_supportInactives = 1. - _predictions_supportInactives
        # predictions = 0.5 * (predictions_supportActives + predictions_supportInactives)
        predictions = predictions_supportActives - _predictions_supportInactives
        # else:
        #    predictions = predictions_supportActives

        predictions = self.sigmoid(self.prediction_scaling * predictions)

        return predictions

    def training_step(self, batch, batch_idx):
        queryMols = batch["queryMolecule"]
        labels = batch["label"]
        supportMolsActive = batch["supportSetActives"]
        supportMolsInactive = batch["supportSetInactives"]
        supportSetActivesSize = batch["supportSetActivesSize"]
        supportSetInactivesSize = batch["supportSetInactivesSize"]
        target_idx = batch["taskIdx"]

        # print(batch_idx)
        # if batch_idx == 65:
        #    import pdb
        #    pdb.set_trace()

        predictions = self.forward(
            queryMols,
            supportMolsActive,
            supportMolsInactive,
            supportSetActivesSize,
            supportSetInactivesSize,
        )
        predictions = torch.squeeze(predictions)

        loss = self.LossFunction(predictions, labels.reshape(-1))

        output = {
            "loss": loss,
            "predictions": predictions,
            "labels": labels,
            "target_idx": target_idx,
        }
        return output

    def validation_step(self, batch, batch_idx):
        queryMols = batch["queryMolecule"]
        labels = batch["label"].squeeze().float()
        supportMolsActive = batch["supportSetActives"]
        supportMolsInactive = batch["supportSetInactives"]
        supportSetActivesSize = batch["supportSetActivesSize"]
        supportSetInactivesSize = batch["supportSetActivesSize"]
        # number_querySet_actives = batch['number_querySet_actives']
        # number_querySet_inactives = batch['number_querySet_inactives']
        target_idx = batch["taskIdx"]

        predictions = self.forward(
            queryMols,
            supportMolsActive,
            supportMolsInactive,
            supportSetActivesSize,
            supportSetInactivesSize,
        ).float()

        loss = self.LossFunction(predictions.reshape(-1), labels)
        # loss = self.LossFunction(predictions[labels!= -1], labels[labels!= -1])

        output = {
            "loss": loss,
            "predictions": predictions,
            "labels": labels,
            "target_idx": target_idx,
        }
        #'number_querySet_actives':number_querySet_actives,
        #'number_querySet_inactives':number_querySet_inactives}
        return output

    def training_epoch_end(self, step_outputs):
        log_dict_training = dict()

        # Predictions
        predictions = torch.cat([x["predictions"] for x in step_outputs], axis=0)
        labels = torch.cat([x["labels"] for x in step_outputs], axis=0)
        epoch_loss = torch.mean(torch.tensor([x["loss"] for x in step_outputs]))
        target_ids = torch.cat([x["target_idx"] for x in step_outputs], axis=0)

        pred_max = torch.max(predictions)
        pred_min = torch.min(predictions)

        auc, _, _ = auc_score_train(predictions.cpu(), labels.cpu(), target_ids.cpu())
        deltaAUPRC, _, _ = deltaAUPRC_score_train(
            predictions.cpu(), labels.cpu(), target_ids.cpu()
        )

        epoch_dict = {
            "loss_train": epoch_loss,
            "auc_train": auc,
            "dAUPRC_train": deltaAUPRC,
            "debug_pred_max_train": pred_max,
            "debug_pred_min_train": pred_min,
        }
        log_dict_training.update(epoch_dict)
        self.log_dict(log_dict_training, "training", on_epoch=True)

    def validation_epoch_end(self, step_outputs):
        log_dict_val = dict()

        # Predictions
        predictions = torch.cat([x["predictions"] for x in step_outputs], axis=0)
        labels = torch.cat([x["labels"] for x in step_outputs], axis=0)
        epoch_loss = torch.mean(torch.tensor([x["loss"] for x in step_outputs]))
        target_ids = torch.cat([x["target_idx"] for x in step_outputs], axis=0)

        auc, _, _ = auc_score_train(predictions.cpu(), labels.cpu(), target_ids.cpu())
        deltaAUPRC, _, _ = deltaAUPRC_score_train(
            predictions.cpu(), labels.cpu(), target_ids.cpu()
        )

        epoch_dict = {"loss_val": epoch_loss, "auc_val": auc, "dAUPRC_val": deltaAUPRC}
        log_dict_val.update(epoch_dict)
        self.log_dict(log_dict_val, "validation", on_epoch=True)

        # epoch_loss_seeds = torch.zeros(5)
        # auc_seeds = np.zeros(5)
        # dauprc_seeds = np.zeros(5)

        # Predictions
        # for dl_idx in range(5):
        #    predictions = torch.cat([x['predictions'] for x in step_outputs[dl_idx]], axis=0)
        #    labels = torch.cat([x['labels'] for x in step_outputs[dl_idx]], axis=0)
        #    epoch_loss = torch.sum(torch.tensor([x["loss"] for x in step_outputs[dl_idx]]))
        #    target_ids = torch.cat([x['target_idx'] for x in step_outputs[dl_idx]], axis=0)
        #    number_querySet_actives = torch.cat([x['number_querySet_actives'] for x in step_outputs[dl_idx]], axis=0)
        #    number_querySet_inactives = torch.cat([x['number_querySet_inactives'] for x in step_outputs[dl_idx]], axis=0)

    def configure_optimizers(self):
        return define_opimizer(self.config, self.parameters())


class IterRef(pl.LightningModule):
    def __init__(self, config: OmegaConf):
        super(IterRef, self).__init__()

        # Config
        self.config = config

        # Loss functions
        self.LossFunction = losses[config.model.training.loss]

        # Hyperparameter
        self.save_hyperparameters(config)

        # Encoder
        self.encoder = EncoderBlock(config)

        # Layernormalizing-block
        if self.config.model.layerNormBlock.usage == True:
            self.layerNormBlock = LayerNormalizingBlock(config)

        # IterRefEmbedding-block
        self.iterRefEmbeddingBlock = IterRefEmbedding(config)

        # Similarity Block
        self.similarity_function = distance_metrics[config.model.similarityBlock.type]

        # Output function
        self.sigmoid = torch.nn.Sigmoid()
        self.prediction_scaling = config.model.prediction_scaling

    def forward(
        self,
        queryMols,
        supportMolsActive,
        supportMolsInactive,
        supportSetActivesSize=0,
        supportSetInactivesSize=0,
    ):
        # Embeddings
        query_embedding = self.encoder(queryMols)
        supportActives_embedding = self.encoder(supportMolsActive)
        supportInactives_embedding = self.encoder(
            supportMolsInactive
        )  # Todo: add if clause below

        # if (self.config.targetSet.numberInactives != 0 or self.config.targetSet.ratioInactives != -1.):
        #    supportInactives_embedding = self.encoder(supportMolsInactive)

        # Layer normalization:
        if self.config.model.layerNormBlock.usage == True:
            # if (self.config.targetSet.numberInactives != 0 or self.config.targetSet.ratioInactives != -1.):
            (
                query_embedding,
                supportActives_embedding,
                supportInactives_embedding,
            ) = self.layerNormBlock(
                query_embedding, supportActives_embedding, supportInactives_embedding
            )
            # else:
            #    (query_embedding, supportActives_embedding,
            #     supportInactives_embedding) = self.layerNormBlock(query_embedding, supportActives_embedding,
            #                                                       None)

        # IterRef
        (
            query_embedding,
            supportActives_embedding,
            supportInactives_embedding,
        ) = self.iterRefEmbeddingBlock(
            query_embedding, supportActives_embedding, supportInactives_embedding
        )

        # Similarities:
        predictions_supportActives = self.similarity_function(
            query_embedding,
            supportActives_embedding,
            supportSetActivesSize,
            device=self.device,
            scaling=self.config.model.similarityBlock.scaling,
            l2Norm=self.config.model.similarityBlock.l2Norm,
        )
        # if (self.config.targetSet.numberInactives != 0 or self.config.targetSet.ratioInactives != -1.):
        _predictions_supportInactives = self.similarity_function(
            query_embedding,
            supportInactives_embedding,
            supportSetInactivesSize,
            device=self.device,
            scaling=self.config.model.similarityBlock.scaling,
            l2Norm=self.config.model.similarityBlock.l2Norm,
        )
        # predictions_supportInactives = 1. - _predictions_supportInactives
        # predictions = 0.5 * (predictions_supportActives + predictions_supportInactives)
        predictions = predictions_supportActives - _predictions_supportInactives
        # else:
        #    predictions = predictions_supportActives

        predictions = self.sigmoid(self.prediction_scaling * predictions)

        return predictions

    def training_step(self, batch, batch_idx):
        queryMols = batch["queryMolecule"]
        labels = batch["label"]
        supportMolsActive = batch["supportSetActives"]
        supportMolsInactive = batch["supportSetInactives"]
        supportSetActivesSize = batch["supportSetActivesSize"]
        supportSetInactivesSize = batch["supportSetInactivesSize"]
        target_idx = batch["taskIdx"]

        predictions = self.forward(
            queryMols,
            supportMolsActive,
            supportMolsInactive,
            supportSetActivesSize,
            supportSetInactivesSize,
        )
        predictions = torch.squeeze(predictions)

        loss = self.LossFunction(predictions, labels.reshape(-1))

        output = {
            "loss": loss,
            "predictions": predictions,
            "labels": labels,
            "target_idx": target_idx,
        }
        return output

    def validation_step(self, batch, batch_idx):
        queryMols = batch["queryMolecule"]
        labels = batch["label"].squeeze().float()
        supportMolsActive = batch["supportSetActives"]
        supportMolsInactive = batch["supportSetInactives"]
        supportSetActivesSize = batch["supportSetActivesSize"]
        supportSetInactivesSize = batch["supportSetActivesSize"]
        # number_querySet_actives = batch['number_querySet_actives']
        # number_querySet_inactives = batch['number_querySet_inactives']
        target_idx = batch["taskIdx"]

        predictions = self.forward(
            queryMols,
            supportMolsActive,
            supportMolsInactive,
            supportSetActivesSize,
            supportSetInactivesSize,
        ).float()

        loss = self.LossFunction(predictions.reshape(-1), labels)
        # loss = self.LossFunction(predictions[labels!= -1], labels[labels!= -1])

        output = {
            "loss": loss,
            "predictions": predictions,
            "labels": labels,
            "target_idx": target_idx,
        }
        #'number_querySet_actives':number_querySet_actives,
        #'number_querySet_inactives':number_querySet_inactives}
        return output

    def training_epoch_end(self, step_outputs):
        log_dict_training = dict()

        # Predictions
        predictions = torch.cat([x["predictions"] for x in step_outputs], axis=0)
        labels = torch.cat([x["labels"] for x in step_outputs], axis=0)
        epoch_loss = torch.mean(torch.tensor([x["loss"] for x in step_outputs]))
        target_ids = torch.cat([x["target_idx"] for x in step_outputs], axis=0)

        pred_max = torch.max(predictions)
        pred_min = torch.min(predictions)

        auc, _, _ = auc_score_train(predictions.cpu(), labels.cpu(), target_ids.cpu())
        deltaAUPRC, _, _ = deltaAUPRC_score_train(
            predictions.cpu(), labels.cpu(), target_ids.cpu()
        )

        epoch_dict = {
            "loss_train": epoch_loss,
            "auc_train": auc,
            "dAUPRC_train": deltaAUPRC,
            "debug_pred_max_train": pred_max,
            "debug_pred_min_train": pred_min,
        }
        log_dict_training.update(epoch_dict)
        self.log_dict(log_dict_training, "training", on_epoch=True)

    def validation_epoch_end(self, step_outputs):
        log_dict_val = dict()

        # Predictions
        predictions = torch.cat([x["predictions"] for x in step_outputs], axis=0)
        labels = torch.cat([x["labels"] for x in step_outputs], axis=0)
        epoch_loss = torch.mean(torch.tensor([x["loss"] for x in step_outputs]))
        target_ids = torch.cat([x["target_idx"] for x in step_outputs], axis=0)

        auc, _, _ = auc_score_train(predictions.cpu(), labels.cpu(), target_ids.cpu())
        deltaAUPRC, _, _ = deltaAUPRC_score_train(
            predictions.cpu(), labels.cpu(), target_ids.cpu()
        )

        epoch_dict = {"loss_val": epoch_loss, "auc_val": auc, "dAUPRC_val": deltaAUPRC}
        log_dict_val.update(epoch_dict)
        self.log_dict(log_dict_val, "validation", on_epoch=True)

        # epoch_loss_seeds = torch.zeros(5)
        # auc_seeds = np.zeros(5)
        # dauprc_seeds = np.zeros(5)

        # Predictions
        # for dl_idx in range(5):
        #    predictions = torch.cat([x['predictions'] for x in step_outputs[dl_idx]], axis=0)
        #    labels = torch.cat([x['labels'] for x in step_outputs[dl_idx]], axis=0)
        #    epoch_loss = torch.sum(torch.tensor([x["loss"] for x in step_outputs[dl_idx]]))
        #    target_ids = torch.cat([x['target_idx'] for x in step_outputs[dl_idx]], axis=0)
        #    number_querySet_actives = torch.cat([x['number_querySet_actives'] for x in step_outputs[dl_idx]], axis=0)
        #    number_querySet_inactives = torch.cat([x['number_querySet_inactives'] for x in step_outputs[dl_idx]], axis=0)

    def configure_optimizers(self):
        return define_opimizer(self.config, self.parameters())
    
class MHNfs(pl.LightningModule):
    def __init__(self, config: OmegaConf):
        super(MHNfs, self).__init__()

        # Config
        self.config = config

        # Load reference set
        self.trainExemplarMemory = (
            torch.unsqueeze(
                torch.from_numpy(
                    np.load(
                        "/system/user/publicdata/FS-Mol/preprocessed_usingQuantils/training/mol_inputs.npy"
                    ).astype("float32")
                ),
                0,
            ).to(config.system.ressources.device)
            # .to("cpu")
        )  #

        self.referenceSet_embedding = torch.ones(1, 512, 1024).to(
            config.system.ressources.device
        )
        # .to("cpu")

        self.layerNorm_refSet = torch.nn.LayerNorm(
            config.model.associationSpace_dim,
            elementwise_affine=config.model.layerNormBlock.affine,
        )

        # Loss functions
        self.LossFunction = losses[config.model.training.loss]

        # Hyperparameter
        self.save_hyperparameters(config)

        # Encoder
        self.encoder = EncoderBlock(config)

        # Hopfield for trained chemical space retrieval
        self.hopfield_chemTrainSpace = HopfieldBlock_chemTrainSpace(self.config)

        # Layernormalizing-block
        if self.config.model.layerNormBlock.usage == True:
            self.layerNormBlock = LayerNormalizingBlock(config)

        # Transformer
        self.transformer = TransformerEmbedding(self.config)

        # Similarity Block
        self.similarity_function = distance_metrics[config.model.similarityBlock.type]

        # Output function
        self.sigmoid = torch.nn.Sigmoid()
        self.prediction_scaling = config.model.prediction_scaling

    def forward(
        self,
        queryMols,
        supportMolsActive,
        supportMolsInactive,
        supportSetActivesSize=0,
        supportSetInactivesSize=0,
    ):
        # Newly added
        if queryMols.dim() == 2:
            queryMols = torch.unsqueeze(queryMols, 0)
        if supportMolsActive.dim() == 2:
            supportMolsActive = torch.unsqueeze(supportMolsActive, 0)
        if supportMolsInactive.dim() == 2:
            supportMolsInactive = torch.unsqueeze(supportMolsInactive, 0)

        # Embeddings
        query_embedding = self.encoder(queryMols)
        supportActives_embedding = self.encoder(supportMolsActive)
        supportInactives_embedding = self.encoder(supportMolsInactive)

        # Retrieve updated representations from chemical training space
        (
            query_embedding,
            supportActives_embedding,
            supportInactives_embedding,
        ) = self.layerNormBlock(
            query_embedding, supportActives_embedding, supportInactives_embedding
        )
        (
            query_embedding,
            supportActives_embedding,
            supportInactives_embedding,
        ) = self.hopfield_chemTrainSpace(
            query_embedding,
            supportActives_embedding,
            supportInactives_embedding,
            supportSetActivesSize,
            supportSetInactivesSize,
            self.referenceSet_embedding,
        )

        # Transformer part
        (
            query_embedding,
            supportActives_embedding,
            supportInactives_embedding,
        ) = self.layerNormBlock(
            query_embedding, supportActives_embedding, supportInactives_embedding
        )
        (
            query_embedding,
            supportActives_embedding,
            supportInactives_embedding,
        ) = self.transformer(
            query_embedding,
            supportActives_embedding,
            supportInactives_embedding,
            supportSetActivesSize,
            supportSetInactivesSize,
        )

        # Layer normalization:
        if self.config.model.layerNormBlock.usage == True:
            (
                query_embedding,
                supportActives_embedding,
                supportInactives_embedding,
            ) = self.layerNormBlock(
                query_embedding, supportActives_embedding, supportInactives_embedding
            )

        # import pdb
        # pdb.set_trace()

        # Similarities:
        predictions_supportActives = self.similarity_function(
            query_embedding,
            supportActives_embedding,
            supportSetActivesSize,
            device=self.device,
            scaling=self.config.model.similarityBlock.scaling,
            l2Norm=self.config.model.similarityBlock.l2Norm,
        )

        _predictions_supportInactives = self.similarity_function(
            query_embedding,
            supportInactives_embedding,
            supportSetInactivesSize,
            device=self.device,
            scaling=self.config.model.similarityBlock.scaling,
            l2Norm=self.config.model.similarityBlock.l2Norm,
        )

        predictions = predictions_supportActives - _predictions_supportInactives

        predictions = self.sigmoid(self.prediction_scaling * predictions)

        return predictions

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            self._update_referenceSet_embedding()

        queryMols = batch["queryMolecule"]
        labels = batch["label"]
        supportMolsActive = batch["supportSetActives"]
        supportMolsInactive = batch["supportSetInactives"]
        supportSetActivesSize = batch["supportSetActivesSize"]
        supportSetInactivesSize = batch["supportSetInactivesSize"]
        target_idx = batch["taskIdx"]

        predictions = self.forward(
            queryMols,
            supportMolsActive,
            supportMolsInactive,
            supportSetActivesSize,
            supportSetInactivesSize,
        )
        predictions = torch.squeeze(predictions)

        loss = self.LossFunction(predictions, labels.reshape(-1))

        output = {
            "loss": loss,
            "predictions": predictions,
            "labels": labels,
            "target_idx": target_idx,
        }
        return output

    def validation_step(self, batch, batch_idx):
        queryMols = batch["queryMolecule"]
        labels = batch["label"].squeeze().float()
        supportMolsActive = batch["supportSetActives"]
        supportMolsInactive = batch["supportSetInactives"]
        supportSetActivesSize = batch["supportSetActivesSize"]
        supportSetInactivesSize = batch["supportSetActivesSize"]
        # number_querySet_actives = batch['number_querySet_actives']
        # number_querySet_inactives = batch['number_querySet_inactives']
        target_idx = batch["taskIdx"]

        predictions = self.forward(
            queryMols,
            supportMolsActive,
            supportMolsInactive,
            supportSetActivesSize,
            supportSetInactivesSize,
        ).float()

        loss = self.LossFunction(predictions.reshape(-1), labels)
        # loss = self.LossFunction(predictions[labels!= -1], labels[labels!= -1])

        output = {
            "loss": loss,
            "predictions": predictions,
            "labels": labels,
            "target_idx": target_idx,
        }
        #'number_querySet_actives':number_querySet_actives,
        #'number_querySet_inactives':number_querySet_inactives}
        return output

    def training_epoch_end(self, step_outputs):
        with torch.no_grad():
            self._update_referenceSet_embedding()

        log_dict_training = dict()

        # Predictions
        predictions = torch.cat([x["predictions"] for x in step_outputs], axis=0)
        labels = torch.cat([x["labels"] for x in step_outputs], axis=0)
        epoch_loss = torch.mean(torch.tensor([x["loss"] for x in step_outputs]))
        target_ids = torch.cat([x["target_idx"] for x in step_outputs], axis=0)

        pred_max = torch.max(predictions)
        pred_min = torch.min(predictions)

        auc, _, _ = auc_score_train(predictions.cpu(), labels.cpu(), target_ids.cpu())
        deltaAUPRC, _, _ = deltaAUPRC_score_train(
            predictions.cpu(), labels.cpu(), target_ids.cpu()
        )

        epoch_dict = {
            "loss_train": epoch_loss,
            "auc_train": auc,
            "dAUPRC_train": deltaAUPRC,
            "debug_pred_max_train": pred_max,
            "debug_pred_min_train": pred_min,
        }
        log_dict_training.update(epoch_dict)
        self.log_dict(log_dict_training, "training", on_epoch=True)

    def validation_epoch_end(self, step_outputs):
        log_dict_val = dict()

        # Predictions
        predictions = torch.cat([x["predictions"] for x in step_outputs], axis=0)
        labels = torch.cat([x["labels"] for x in step_outputs], axis=0)
        epoch_loss = torch.mean(torch.tensor([x["loss"] for x in step_outputs]))
        target_ids = torch.cat([x["target_idx"] for x in step_outputs], axis=0)

        auc, _, _ = auc_score_train(predictions.cpu(), labels.cpu(), target_ids.cpu())
        deltaAUPRC, _, _ = deltaAUPRC_score_train(
            predictions.cpu(), labels.cpu(), target_ids.cpu()
        )

        epoch_dict = {"loss_val": epoch_loss, "auc_val": auc, "dAUPRC_val": deltaAUPRC}
        log_dict_val.update(epoch_dict)
        self.log_dict(log_dict_val, "validation", on_epoch=True)

    def configure_optimizers(self):
        return define_opimizer(self.config, self.parameters())

    def _update_referenceSet_embedding(self):
        max_rows = self.trainExemplarMemory.shape[1]
        number_requested_rows = int(np.rint(0.05 * max_rows))

        sampled_rows = torch.randperm(max_rows)[:number_requested_rows]

        self.referenceSet_embedding = self.layerNorm_refSet(
            self.encoder(self.trainExemplarMemory[:, sampled_rows, :])
        )

    def _get_retrival_and_referenceembedding(
        self,
        queryMols,
        supportMolsActive,
        supportMolsInactive,
        supportSetActivesSize=0,
        supportSetInactivesSize=0,
    ):
        # Embeddings
        query_embedding_init = self.encoder(queryMols)
        supportActives_embedding_init = self.encoder(supportMolsActive)
        supportInactives_embedding_init = self.encoder(supportMolsInactive)

        # LayerNorm
        (
            query_embedding_init,
            supportActives_embedding_init,
            supportInactives_embedding_init,
        ) = self.layerNormBlock(
            query_embedding_init,
            supportActives_embedding_init,
            supportInactives_embedding_init,
        )

        # Retrieve updated representations from chemical training space
        (
            query_embedding,
            supportActives_embedding,
            supportInactives_embedding,
        ) = self.hopfield_chemTrainSpace(
            query_embedding_init,
            supportActives_embedding_init,
            supportInactives_embedding_init,
            supportSetActivesSize,
            supportSetInactivesSize,
            self.referenceSet_embedding,
        )

        (
            query_embedding,
            supportActives_embedding,
            supportInactives_embedding,
        ) = self.layerNormBlock(
            query_embedding, supportActives_embedding, supportInactives_embedding
        )

        return (
            query_embedding_init,
            supportActives_embedding_init,
            supportInactives_embedding_init,
            query_embedding,
            supportActives_embedding,
            supportInactives_embedding,
            self.referenceSet_embedding,
        )

    def _get_hopfield_association_mtx(
        self, queryMols, supportMolsActive, supportMolsInactive
    ):
        # Embeddings
        query_embedding_init = self.encoder(queryMols)
        supportActives_embedding_init = self.encoder(supportMolsActive)
        supportInactives_embedding_init = self.encoder(supportMolsInactive)

        # LayerNorm
        (
            query_embedding_init,
            supportActives_embedding_init,
            supportInactives_embedding_init,
        ) = self.layerNormBlock(
            query_embedding_init,
            supportActives_embedding_init,
            supportInactives_embedding_init,
        )

        # Retrieve updated representations from chemical training space
        S = torch.cat(
            (
                query_embedding_init,
                supportActives_embedding_init,
                supportInactives_embedding_init,
            ),
            1,
        )

        S_flattend = S.reshape(1, S.shape[0] * S.shape[1], S.shape[2])

        association_mtx = self.hopfield_chemTrainSpace.hopfield.get_association_matrix(
            (self.referenceSet_embedding, S_flattend, self.referenceSet_embedding)
        )

        return association_mtx

    def _get_crossAttentionEmbeddings(
        self,
        queryMols,
        supportMolsActive,
        supportMolsInactive,
        supportSetActivesSize=0,
        supportSetInactivesSize=0,
    ):
        # Embeddings
        query_embedding = self.encoder(queryMols)
        supportActives_embedding = self.encoder(supportMolsActive)
        supportInactives_embedding = self.encoder(supportMolsInactive)

        # LayerNorm
        (
            query_embedding,
            supportActives_embedding,
            supportInactives_embedding,
        ) = self.layerNormBlock(
            query_embedding, supportActives_embedding, supportInactives_embedding
        )

        # Retrieve updated representations from chemical training space
        (
            query_embedding,
            supportActives_embedding,
            supportInactives_embedding,
        ) = self.hopfield_chemTrainSpace(
            query_embedding,
            supportActives_embedding,
            supportInactives_embedding,
            supportSetActivesSize,
            supportSetInactivesSize,
            self.referenceSet_embedding,
        )

        (
            query_embedding_input,
            supportActives_embedding_input,
            supportInactives_embedding_input,
        ) = self.layerNormBlock(
            query_embedding, supportActives_embedding, supportInactives_embedding
        )

        # Cross Attention module
        (
            query_embedding,
            supportActives_embedding,
            supportInactives_embedding,
        ) = self.transformer(
            query_embedding_input,
            supportActives_embedding_input,
            supportInactives_embedding_input,
            supportSetActivesSize,
            supportSetInactivesSize,
        )

        (
            query_embedding,
            supportActives_embedding,
            supportInactives_embedding,
        ) = self.layerNormBlock(
            query_embedding, supportActives_embedding, supportInactives_embedding
        )

        return (
            query_embedding_input,
            supportActives_embedding_input,
            supportInactives_embedding_input,
            query_embedding,
            supportActives_embedding,
            supportInactives_embedding,
        )

    def _get_retrival_and_referenceembedding_justhopfieldblock(
        self,
        queryMols,
        supportMolsActive,
        supportMolsInactive,
        supportSetActivesSize=0,
        supportSetInactivesSize=0,
    ):
        # Retrieve updated representations from chemical training space
        (query_embedding, _, _) = self.hopfield_chemTrainSpace(
            queryMols,
            supportMolsActive,
            supportMolsInactive,
            supportSetActivesSize,
            supportSetInactivesSize,
            self.referenceSet_embedding,
        )
        return query_embedding, self.referenceSet_embedding


