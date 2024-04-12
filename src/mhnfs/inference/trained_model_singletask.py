"""
This file includes all necessary code to build the MHNfs model. This code is used to
RUN the model during inference time for a single target.

This means:
- Padding is turned off:
    * The support set inputs are not padded. This allows for abitrary support set sizes.
      Multiple support sets are not supported though.
"""

import pytorch_lightning as pl
import numpy as np
import torch

from src.mhnfs.inference.modules_trained_mhnfs import (
    EncoderBlock,
    ContextModule,
    LayerNormalizingBlock,
    CrossAttentionModule,
    SimilarityModule,
)

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
        current_loc = __file__.rsplit("/",4)[0]
        self.context = (
            torch.unsqueeze(
                torch.from_numpy(
                    np.load(current_loc + "/assets/mhnfs_data/full_context_set.npy")
                ),
                0,
            )
            .float()
        )

        self.context_embedding = torch.ones(1, 512, 1024)

        self.layerNorm_context = torch.nn.LayerNorm(
            cfg.model.associationSpace_dim,
            elementwise_affine=cfg.model.layerNormBlock.affine,
        )

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

        # - Expand support set related tensors
        support_actives_embedding = support_actives_embedding.expand(
                                                    query_embedding.shape[0], -1, -1)
        support_inactives_embedding = support_inactives_embedding.expand(
                                                    query_embedding.shape[0], -1, -1)
        support_set_actives_size = support_set_actives_size.expand(
                                                    query_embedding.shape[0])
        support_set_inactives_size = support_set_inactives_size.expand(
                                                    query_embedding.shape[0])
        
        # - Context module
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.contextModule(
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
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
            support_set_actives_size,
            self.cfg,
        )

        predictions_support_inactives = self.similarity_function(
            query_embedding,
            support_inactives_embedding,
            support_set_inactives_size,
            self.cfg,
        )

        predictions = predictions_support_actives - predictions_support_inactives

        predictions = self.sigmoid(self.prediction_scaling * predictions)

        return predictions

    @torch.no_grad()
    def _update_context_set_embedding(self):
        """
        This function randomly samples the context set as a subset of all available
        training molecules
        """
        max_rows = self.context.shape[1]
        number_requested_rows = int(
            np.rint(self.cfg.model.context.ratio_training_molecules * max_rows)
        )

        sampled_rows = torch.randperm(max_rows)[:number_requested_rows]

        self.context_embedding = self.layerNorm_context(
            self.encoder(self.context[:, sampled_rows, :])
        )
