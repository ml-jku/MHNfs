import pytorch_lightning as pl
import numpy as np
import torch
import hydra

from src.mhnfs.modules import (
    EncoderBlock,
    ContextModule,
    LayerNormalizingBlock,
    CrossAttentionModule,
    SimilarityModule,
)

from src.metrics.performance_metrics import (compute_auc_score,
                                             compute_dauprc_score)
from src.metrics.tracker import (MovingAverageTracker,
                                 TrainValDeltaTracker)
from src.mhnfs.optimizer import define_opimizer


class MHNfs(pl.LightningModule):
    """
    The MHNfs is a few-shot drug-discovery model for activity prediction.

    For a requested query molecule, MHNfs predicts activity, while known knowledge from
    the support set is used.

    MHNfs can be seen as an embedding-based few-shot method since the prediction is
    based on similarities of molecule representations in a learned "representation
    space". Being able to build rich, expressive molecule representations is the key for
    a predictive model.

    MHNfs consists of
    three consecutive modules:
    - the context module,
    - the cross attention module, and
    - the similarity module.

    The context module associates the query and support set molecules with context -
    i.e., a large set of training molecules.

    The cross-attention module allows for information sharing between query and support
    set molecules.

    The similirity modules computes pair-wise similarity values between query and sup-
    port set molecules and uses these similarity values to build a prediction from a
    weighted sum over the support set labels.
    """

    def __init__(self, cfg):
        super(MHNfs, self).__init__()

        # Config
        self.cfg = cfg

        # Load context set
        self.context = (
            torch.unsqueeze(
                torch.from_numpy(
                    np.load(
                        cfg.system.data.path
                        + cfg.system.data.dir_training
                        + cfg.system.data.name_mol_inputs
                    )
                ),
                0,
            )
            .float()
            .to(cfg.system.ressources.device) # .to('cpu') #
        )

        self.context_embedding = torch.ones(1, 512, 1024
                                            ).to(cfg.system.ressources.device) # .to('cpu') #

        self.layerNorm_context = torch.nn.LayerNorm(
            cfg.model.associationSpace_dim,
            elementwise_affine=cfg.model.layerNormBlock.affine,
        )

        # Loss functions
        self.lossFunction = torch.nn.BCELoss()
        
        # Moving average and train-val-delta tracker
        self.val_dauprc_ma_tracker = MovingAverageTracker()
        self.train_val_delta_tracker = TrainValDeltaTracker()
        
        # Hyperparameters
        self.save_hyperparameters(cfg)

        # Encoder
        self.encoder = EncoderBlock(cfg)

        # Context module
        self.contextModule = ContextModule(self.cfg)

        # Layernormalizing-block
        self.layerNormBlock = LayerNormalizingBlock(cfg)

        # Cross-attention module
        self.crossAttentionModule = CrossAttentionModule(self.cfg)

        # Similarity module
        self.similarity_function = SimilarityModule

        # Output function
        self.sigmoid = torch.nn.Sigmoid()
        self.prediction_scaling = cfg.model.prediction_scaling

    def forward(
        self,
        query_molecules: torch.Tensor,
        support_molecules_active: torch.Tensor,
        support_molecules_inactive: torch.Tensor,
        support_set_actives_size: torch.Tensor,
        support_set_inactives_size: torch.Tensor,
        support_molecules_active_mask: torch.Tensor,
        support_molecules_inactive_mask: torch.Tensor,
    ) -> torch.Tensor:
        
        # Get embeddings from molecule encoder
        query_embedding = self.encoder(query_molecules)
        support_actives_embedding = self.encoder(support_molecules_active)
        support_inactives_embedding = self.encoder(support_molecules_inactive)

        # Retrieve updated representations from the context module
        # - Layernorm
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.layerNormBlock(
            query_embedding, support_actives_embedding, support_inactives_embedding
        )

        # - Context module
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.contextModule(
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
            # support_set_actives_size,
            # support_set_inactives_size,
            self.context_embedding,
        )

        # Allow for information sharing between query and support set
        # - Layernorm
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.layerNormBlock(
            query_embedding, support_actives_embedding, support_inactives_embedding
        )

        # - Cross-attention module
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.crossAttentionModule(
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
            support_molecules_active_mask,
            support_molecules_inactive_mask,
        )

        # Build predictions from a weighted sum over support set labels
        # - Layernorm:
        if self.cfg.model.layerNormBlock.usage:
            (
                query_embedding,
                support_actives_embedding,
                support_inactives_embedding,
            ) = self.layerNormBlock(
                query_embedding, support_actives_embedding, support_inactives_embedding
            )

        # - Similarity module:
        predictions_support_actives = self.similarity_function(
            query_embedding,
            support_actives_embedding,
            support_molecules_active_mask,
            support_set_actives_size,
            self.cfg,
        )

        predictions_support_inactives = self.similarity_function(
            query_embedding,
            support_inactives_embedding,
            support_molecules_inactive_mask,
            support_set_inactives_size,
            self.cfg,
        )

        predictions = predictions_support_actives - predictions_support_inactives

        predictions = self.sigmoid(self.prediction_scaling * predictions)

        return predictions

    def training_step(self, batch, batch_idx):
        # Update context molecules
        self._update_context_set_embedding()

        query_molecules = batch["queryMolecule"]
        labels = batch["label"].float()
        support_molecules_active = batch["supportSetActives"]
        support_molecules_inactive = batch["supportSetInactives"]
        support_set_actives_size = batch["supportSetActivesSize"]
        support_set_inactives_size = batch["supportSetInactivesSize"]
        target_idx = batch["taskIdx"]
        support_molecules_active_mask = torch.cat(
            [
                torch.cat(
                    [torch.tensor([False] * d), torch.tensor([True] * (12 - d))]
                ).reshape(1, -1)
                for d in support_set_actives_size
            ],
            dim=0,
        ).to(self.cfg.system.ressources.device)
        
        
        support_molecules_inactive_mask = torch.cat(
            [
                torch.cat(
                    [torch.tensor([False] * d), torch.tensor([True] * (12 - d))]
                ).reshape(1, -1)
                for d in support_set_inactives_size
            ],
            dim=0,
        ).to(self.cfg.system.ressources.device)
        
        # Input dropout for support set
        
        # For variations
        #thr = np.random.uniform(0, self.cfg.model.transformer.ss_dropout)
        
        active_mask_dropout = torch.rand(support_molecules_active_mask.shape).to(
            self.cfg.system.ressources.device) < self.cfg.model.transformer.ss_dropout
            #self.cfg.system.ressources.device) < thr
        support_molecules_active_mask = torch.logical_or(support_molecules_active_mask,
                                                         active_mask_dropout)
        
        inactive_mask_dropout = torch.rand(support_molecules_inactive_mask.shape).to(
            self.cfg.system.ressources.device) < self.cfg.model.transformer.ss_dropout
        support_molecules_inactive_mask = torch.logical_or(
            support_molecules_inactive_mask, inactive_mask_dropout)
        

        predictions = self.forward(
            query_molecules,
            support_molecules_active,
            support_molecules_inactive,
            support_set_actives_size,
            support_set_inactives_size,
            support_molecules_active_mask,
            support_molecules_inactive_mask,
        )
        predictions = torch.squeeze(predictions)

        loss = self.lossFunction(predictions, labels.reshape(-1))

        output = {
            "loss": loss,
            "predictions": predictions,
            "labels": labels,
            "target_idx": target_idx,
        }
        return output

    def validation_step(self, batch, batch_idx):
        query_molecules = batch["queryMolecule"]
        labels = batch["label"].float()
        support_molecules_active = batch["supportSetActives"]
        support_molecules_inactive = batch["supportSetInactives"]
        support_set_actives_size = batch["supportSetActivesSize"]
        support_set_inactives_size = batch["supportSetInactivesSize"]
        target_idx = batch["taskIdx"]
        support_molecules_active_mask = torch.cat(
            [
                torch.cat(
                    [torch.tensor([False] * d), torch.tensor([True] * (12 - d))]
                ).reshape(1, -1)
                for d in support_set_actives_size
            ],
            dim=0,
        ).to(self.cfg.system.ressources.device)
        
        support_molecules_inactive_mask = torch.cat(
            [
                torch.cat(
                    [torch.tensor([False] * d), torch.tensor([True] * (12 - d))]
                ).reshape(1, -1)
                for d in support_set_inactives_size
            ],
            dim=0,
        ).to(self.cfg.system.ressources.device)

        predictions = self.forward(
            query_molecules,
            support_molecules_active,
            support_molecules_inactive,
            support_set_actives_size,
            support_set_inactives_size,
            support_molecules_active_mask,
            support_molecules_inactive_mask,
        ).float()

        loss = self.lossFunction(predictions.reshape(-1), labels)

        output = {
            "loss": loss,
            "predictions": predictions,
            "labels": labels,
            "target_idx": target_idx,
        }
        return output

    def training_epoch_end(self, step_outputs):
        with torch.no_grad():
            self._update_context_set_embedding()

        log_dict_training = dict()

        # Predictions
        predictions = torch.cat([x["predictions"] for x in step_outputs], dim=0)
        labels = torch.cat([x["labels"] for x in step_outputs], dim=0)
        epoch_loss = torch.mean(torch.tensor([x["loss"] for x in step_outputs]))
        target_ids = torch.cat([x["target_idx"] for x in step_outputs], dim=0)

        auc, _, _ = compute_auc_score(predictions.cpu(), labels.cpu(), target_ids.cpu())
        dauprc, _, _ = compute_dauprc_score(
            predictions.cpu(), labels.cpu(), target_ids.cpu()
        )
        
        # Udpdate train-val-delta tracker
        self.train_val_delta_tracker.set_train_value(dauprc)

        epoch_dict = {
            "loss_train": epoch_loss,
            "auc_train": auc,
            "dAUPRC_train": dauprc,
        }
        log_dict_training.update(epoch_dict)
        self.log_dict(log_dict_training, on_epoch=True)

    def validation_epoch_end(self, step_outputs):
        log_dict_val = dict()

        # Predictions
        predictions = torch.cat([x["predictions"] for x in step_outputs], dim=0)
        labels = torch.cat([x["labels"] for x in step_outputs], dim=0)
        epoch_loss = torch.mean(torch.tensor([x["loss"] for x in step_outputs]))
        target_ids = torch.cat([x["target_idx"] for x in step_outputs], dim=0)

        auc, _, _ = compute_auc_score(predictions.cpu(), labels.cpu(), target_ids.cpu())
        dauprc, _, _ = compute_dauprc_score(
            predictions.cpu(), labels.cpu(), target_ids.cpu()
        )
        
        # Update moving average tracker
        self.val_dauprc_ma_tracker.update(dauprc)
        # Udpdate train-val-delta tracker
        self.train_val_delta_tracker.set_val_value(dauprc)
        dauprc_train_val_delta = self.train_val_delta_tracker.absolute_delta

        epoch_dict = {"loss_val": epoch_loss, "auc_val": auc, "dAUPRC_val": dauprc,
                      "dAUPRC_val_ma": self.val_dauprc_ma_tracker.value,
                      "dAUPRC_train_val_delta": dauprc_train_val_delta}
        log_dict_val.update(epoch_dict)
        self.log_dict(log_dict_val, on_epoch=True)

    def configure_optimizers(self):
        return define_opimizer(self.cfg, self.parameters())

    @torch.no_grad()
    def _update_context_set_embedding(self):
        """
        todo describe, random choice, ....
        """
        max_rows = self.context.shape[1]
        number_requested_rows = int(
            np.rint(self.cfg.model.context.ratio_training_molecules * max_rows)
        )

        sampled_rows = torch.randperm(max_rows)[:number_requested_rows]

        self.context_embedding = self.layerNorm_context(
            self.encoder(self.context[:, sampled_rows, :])
        )

    @torch.no_grad()
    def _get_context_retrieval(
        self,
        query_molecules,
        support_molecules_active,
        support_molecules_inactive,
        support_set_actives_size,
        support_set_inactives_size,
    ):
        # Embeddings
        query_embedding_init = self.encoder(query_molecules)
        support_actives_embedding_init = self.encoder(support_molecules_active)
        support_inactives_embedding_init = self.encoder(support_molecules_inactive)

        # LayerNorm
        (
            query_embedding_init,
            support_actives_embedding_init,
            support_inactives_embedding_init,
        ) = self.layerNormBlock(
            query_embedding_init,
            support_actives_embedding_init,
            support_inactives_embedding_init,
        )

        # Retrieve updated representations from chemical training space
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.contextModule(
            query_embedding_init,
            support_actives_embedding_init,
            support_inactives_embedding_init,
            support_set_actives_size,
            support_set_inactives_size,
            self.context_embedding,
        )

        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.layerNormBlock(
            query_embedding, support_actives_embedding, support_inactives_embedding
        )

        return (
            query_embedding_init,
            support_actives_embedding_init,
            support_inactives_embedding_init,
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
            self.context_embedding,
        )

    @torch.no_grad()
    def _get_hopfield_association_mtx(
        self, query_molecules, support_molecules_active, support_molecules_inactive
    ):
        # Embeddings
        query_embedding_init = self.encoder(query_molecules)
        support_actives_embedding_init = self.encoder(support_molecules_active)
        support_inactives_embedding_init = self.encoder(support_molecules_inactive)

        # LayerNorm
        (
            query_embedding_init,
            supportActives_embedding_init,
            supportInactives_embedding_init,
        ) = self.layerNormBlock(
            query_embedding_init,
            support_actives_embedding_init,
            support_inactives_embedding_init,
        )

        # Retrieve updated representations from chemical training space
        s = torch.cat(
            (
                query_embedding_init,
                supportActives_embedding_init,
                supportInactives_embedding_init,
            ),
            1,
        )

        s_flattend = s.reshape(1, s.shape[0] * s.shape[1], s.shape[2])

        association_mtx = self.contextModule.hopfield.get_association_matrix(
            (self.context_embedding, s_flattend, self.context_embedding)
        )

        return association_mtx

    @torch.no_grad()
    def _get_cross_attention_embeddings(
        self,
        query_molecules,
        support_molecules_active,
        support_molecules_inactive,
        support_set_actives_size=0,
        support_set_inactives_size=0,
    ):
        # Embeddings
        query_embedding = self.encoder(query_molecules)
        support_actives_embedding = self.encoder(support_molecules_active)
        support_inactives_embedding = self.encoder(support_molecules_inactive)

        # LayerNorm
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.layerNormBlock(
            query_embedding, support_actives_embedding, support_inactives_embedding
        )

        # Retrieve updated representations from context
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.contextModule(
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
            support_set_actives_size,
            support_set_inactives_size,
            self.context_embedding,
        )
        (
            query_embedding_input,
            support_actives_embedding_input,
            support_inactives_embedding_input,
        ) = self.layerNormBlock(
            query_embedding, support_actives_embedding, support_inactives_embedding
        )

        # Cross Attention module
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.crossAttentionModule(
            query_embedding_input,
            support_actives_embedding_input,
            support_inactives_embedding_input,
            support_set_actives_size,
            support_set_inactives_size,
        )
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.layerNormBlock(
            query_embedding, support_actives_embedding, support_inactives_embedding
        )

        return (
            query_embedding_input,
            support_actives_embedding_input,
            support_inactives_embedding_input,
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        )


# --------------------------------------------------------------------------------------
# For debugging:
# --------------------------------------------------------------------------------------


@hydra.main(config_path="../configs", config_name="cfg")
def debug_mhnfs(cfg):
    model = MHNfs(cfg).to("cuda")

    query_molecules = torch.rand(10, 1, 2248).to("cuda")
    labels = torch.rand(10).to("cuda")
    support_molecules_active = torch.rand(10, 8, 2248).to("cuda")
    support_molecules_inactive = torch.rand(10, 8, 2248).to("cuda")
    support_set_actives_size = torch.rand(10).to("cuda")
    support_set_inactives_size = torch.rand(10).to("cuda")
    support_molecules_active_mask = torch.randint(0, 2, (10, 8)).bool().to("cuda")
    support_molecules_inactive_mask = torch.randint(0, 2, (10, 8)).bool().to("cuda")

    preds = model(
        query_molecules,
        support_molecules_active,
        support_molecules_inactive,
        support_set_actives_size,
        support_set_inactives_size,
        support_molecules_active_mask,
        support_molecules_inactive_mask,
    ).float()
    print(preds)

    loss = model.lossFunction(preds.reshape(-1), labels)
    print(loss)


if __name__ == "__main__":
    debug_mhnfs()
