import pytorch_lightning as pl
import numpy as np
import torch

from modules import EncoderBlock, ContextModule, LayerNormalizingBlock, CrossAttentionModule
from modules import similarity_module
from metrics import compute_auc_score, compute_dauprc_score
from optimizer import define_opimizer


class MHNfs(pl.LightningModule):
    def __init__(self, config):
        super(MHNfs, self).__init__()

        # Config
        self.config = config

        # Load context set
        self.context = torch.unsqueeze(torch.from_numpy(
            np.load(config.system.path + config.system.dir_training + config.system.name_mol_inputs)
        ), 0).to(config.system.ressources.device)

        self.context_embedding = torch.ones(1, 512, 1024).to(config.system.ressources.device)

        self.layerNorm_context = torch.nn.LayerNorm(config.model.associationSpace_dim,
                                                    elementwise_affine=config.model.layerNormBlock.affine)

        # Loss functions
        self.lossFunction = torch.nn.BCELoss()

        # Hyperparameters
        self.save_hyperparameters(config)

        # Encoder
        self.encoder = EncoderBlock(config)

        # Context module
        self.contextModule = ContextModule(self.config)

        # Layernormalizing-block
        if self.config.model.layerNormBlock.usage:
            self.layerNormBlock = LayerNormalizingBlock(config)

        # Cross-attention module
        self.crossAttentionModule = CrossAttentionModule(self.config)

        # Similarity module
        self.similarity_function = similarity_module

        # Output function
        self.sigmoid = torch.nn.Sigmoid()
        self.prediction_scaling = config.model.prediction_scaling

    def forward(self, query_molecules, support_molecules_active, support_molecules_inactive,
                support_set_actives_size=0, support_set_inactives_size=0):
        # Embeddings
        query_embedding = self.encoder(query_molecules)
        support_actives_embedding = self.encoder(support_molecules_active)
        support_inactives_embedding = self.encoder(support_molecules_inactive)

        # Retrieve updated representations from the context module
        (
            query_embedding, support_actives_embedding, support_inactives_embedding
        ) = self.layerNormBlock(query_embedding, support_actives_embedding, support_inactives_embedding)
        (
            query_embedding, support_actives_embedding, support_inactives_embedding
        ) = self.contextModule(query_embedding, support_actives_embedding, support_inactives_embedding,
                               support_set_actives_size, support_set_inactives_size, self.referenceSet_embedding)

        # Cross-attention module
        (
            query_embedding, support_actives_embedding, support_inactives_embedding
        ) = self.layerNormBlock(query_embedding, support_actives_embedding, support_inactives_embedding)
        (
            query_embedding, support_actives_embedding, support_inactives_embedding
        ) = self.crossAttentionModule(query_embedding, support_actives_embedding, support_inactives_embedding,
                                      support_set_actives_size, support_set_inactives_size)

        # Layer normalization:
        if self.config.model.layerNormBlock.usage:
            (
                query_embedding, support_actives_embedding, support_inactives_embedding
            ) = self.layerNormBlock(query_embedding, support_actives_embedding, support_inactives_embedding)

        # Similarities:
        predictions_support_actives = self.similarity_function(query_embedding, support_actives_embedding,
                                                               support_set_actives_size,
                                                               self.config)

        predictions_support_inactives = self.similarity_function(query_embedding, support_inactives_embedding,
                                                                 support_set_actives_size,
                                                                 self.config)

        predictions = predictions_support_actives - predictions_support_inactives

        predictions = self.sigmoid(self.prediction_scaling * predictions)

        return predictions

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            self._update_referenceSet_embedding()

        query_molecules = batch['queryMolecule']
        labels = batch['label']
        support_molecules_active = batch['supportSetActives']
        support_molecules_inactive = batch['supportSetInactives']
        support_set_actives_size = batch['support_set_actives_size']
        support_set_inactives_size = batch['support_set_inactives_size']
        target_idx = batch['taskIdx']

        predictions = self.forward(query_molecules, support_molecules_active, support_molecules_inactive,
                                   support_set_actives_size, support_set_inactives_size)
        predictions = torch.squeeze(predictions)

        loss = self.lossFunction(predictions, labels.reshape(-1))

        output = {'loss': loss, 'predictions': predictions, 'labels': labels, 'target_idx': target_idx}
        return output

    def validation_step(self, batch, batch_idx):
        query_molecules = batch['queryMolecule']
        labels = batch['label'].squeeze().float()
        support_molecules_active = batch['supportSetActives']
        support_molecules_inactive = batch['supportSetInactives']
        support_set_actives_size = batch['support_set_actives_size']
        support_set_inactives_size = batch['support_set_actives_size']
        target_idx = batch['taskIdx']

        predictions = self.forward(query_molecules, support_molecules_active, support_molecules_inactive,
                                   support_set_actives_size, support_set_inactives_size).float()

        loss = self.lossFunction(predictions.reshape(-1), labels)

        output = {'loss': loss, 'predictions': predictions, 'labels': labels, 'target_idx': target_idx}
        return output

    def training_epoch_end(self, step_outputs):
        with torch.no_grad():
            self._update_referenceSet_embedding()

        log_dict_training = dict()

        # Predictions
        predictions = torch.cat([x['predictions'] for x in step_outputs], dim=0)
        labels = torch.cat([x['labels'] for x in step_outputs], dim=0)
        epoch_loss = torch.mean(torch.tensor([x["loss"] for x in step_outputs]))
        target_ids = torch.cat([x['target_idx'] for x in step_outputs], dim=0)

        auc, _, _ = compute_auc_score(predictions.cpu(), labels.cpu(), target_ids.cpu())
        dauprc, _, _ = compute_dauprc_score(predictions.cpu(), labels.cpu(), target_ids.cpu())

        epoch_dict = {'loss_train': epoch_loss, 'auc_train': auc, 'dAUPRC_train': dauprc}
        log_dict_training.update(epoch_dict)
        self.log_dict(log_dict_training, on_epoch=True)

    def validation_epoch_end(self, step_outputs):
        log_dict_val = dict()

        # Predictions
        predictions = torch.cat([x['predictions'] for x in step_outputs], dim=0)
        labels = torch.cat([x['labels'] for x in step_outputs], dim=0)
        epoch_loss = torch.mean(torch.tensor([x["loss"] for x in step_outputs]))
        target_ids = torch.cat([x['target_idx'] for x in step_outputs], dim=0)

        auc, _, _ = compute_auc_score(predictions.cpu(), labels.cpu(), target_ids.cpu())
        dauprc, _, _ = compute_dauprc_score(predictions.cpu(), labels.cpu(), target_ids.cpu())

        epoch_dict = {'loss_val': epoch_loss, 'auc_val': auc, 'dAUPRC_val': dauprc}
        log_dict_val.update(epoch_dict)
        self.log_dict(log_dict_val, on_epoch=True)

    def configure_optimizers(self):
        return define_opimizer(self.config, self.parameters())

    def _update_reference_set_embedding(self):
        max_rows = self.context.shape[1]
        number_requested_rows = int(np.rint(self.config.model.context.ratio_training_molecules * max_rows))

        sampled_rows = torch.randperm(max_rows)[:number_requested_rows]

        self.referenceSet_embedding = self.layerNorm_context(
            self.encoder(self.context[:, sampled_rows, :])
        )

    def _get_context_retrieval(self, query_molecules, support_molecules_active, support_molecules_inactive,
                               support_set_actives_size, support_set_inactives_size):
        # Embeddings
        query_embedding_init = self.encoder(query_molecules)
        support_actives_embedding_init = self.encoder(support_molecules_active)
        support_inactives_embedding_init = self.encoder(support_molecules_inactive)

        # LayerNorm
        (
            query_embedding_init, support_actives_embedding_init, support_inactives_embedding_init
        ) = self.layerNormBlock(query_embedding_init, support_actives_embedding_init, support_inactives_embedding_init)

        # Retrieve updated representations from chemical training space
        (
            query_embedding, support_actives_embedding, support_inactives_embedding
        ) = self.contextModule(query_embedding_init, support_actives_embedding_init, support_inactives_embedding_init,
                               support_set_actives_size, support_set_inactives_size, self.context_embedding)

        (
            query_embedding, support_actives_embedding, support_inactives_embedding
        ) = self.layerNormBlock(query_embedding, support_actives_embedding, support_inactives_embedding)

        return (query_embedding_init, support_actives_embedding_init, support_inactives_embedding_init,
                query_embedding, support_actives_embedding, support_inactives_embedding, self.referenceSet_embedding)

    def _get_hopfield_association_mtx(self, query_molecules, support_molecules_active, support_molecules_inactive):
        # Embeddings
        query_embedding_init = self.encoder(query_molecules)
        support_actives_embedding_init = self.encoder(support_molecules_active)
        support_inactives_embedding_init = self.encoder(support_molecules_inactive)

        # LayerNorm
        (
            query_embedding_init, supportActives_embedding_init, supportInactives_embedding_init
        ) = self.layerNormBlock(query_embedding_init, support_actives_embedding_init, support_inactives_embedding_init)

        # Retrieve updated representations from chemical training space
        s = torch.cat((query_embedding_init, supportActives_embedding_init, supportInactives_embedding_init), 1)

        s_flattend = s.reshape(1, s.shape[0] * s.shape[1], s.shape[2])

        association_mtx = self.contextModule.hopfield.get_association_matrix(
            (self.context_embedding, s_flattend, self.context_embedding)
        )

        return association_mtx

    def _get_cross_attention_embeddings(self, query_molecules, support_molecules_active, support_molecules_inactive,
                                        support_set_actives_size=0, support_set_inactives_size=0):

        # Embeddings
        query_embedding = self.encoder(query_molecules)
        support_actives_embedding = self.encoder(support_molecules_active)
        support_inactives_embedding = self.encoder(support_molecules_inactive)

        # LayerNorm
        (
            query_embedding, support_actives_embedding, support_inactives_embedding
        ) = self.layerNormBlock(query_embedding, support_actives_embedding, support_inactives_embedding)

        # Retrieve updated representations from context
        (
            query_embedding, support_actives_embedding, support_inactives_embedding
        ) = self.contextModule(query_embedding, support_actives_embedding, support_inactives_embedding,
                               support_set_actives_size, support_set_inactives_size, self.context_embedding)
        (
            query_embedding_input, support_actives_embedding_input, support_inactives_embedding_input
        ) = self.layerNormBlock(query_embedding, support_actives_embedding, support_inactives_embedding)

        # Cross Attention module
        (
            query_embedding, support_actives_embedding, support_inactives_embedding
        ) = self.crossAttentionModule(query_embedding_input,
                                      support_actives_embedding_input, support_inactives_embedding_input,
                                      support_set_actives_size, support_set_inactives_size)
        (
            query_embedding, support_actives_embedding, support_inactives_embedding
        ) = self.layerNormBlock(query_embedding, support_actives_embedding, support_inactives_embedding)

        return (query_embedding_input, support_actives_embedding_input, support_inactives_embedding_input,
                query_embedding, support_actives_embedding, support_inactives_embedding)
