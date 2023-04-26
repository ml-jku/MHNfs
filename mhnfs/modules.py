import torch
import torch.nn as nn
from omegaconf import OmegaConf
from functools import partial

from hopfield.modules import Hopfield
from initialization import init_weights


# Mappings
activation_function_mapping = {
    'relu': nn.ReLU(),
    'selu': nn.SELU(),
    'sigmoid': nn.Sigmoid()
}

dropout_mapping = {
    'relu': nn.Dropout,
    'selu': nn.AlphaDropout
}


# Modules
class EncoderBlock(nn.Module):
    def __init__(self, config: OmegaConf):
        super(EncoderBlock, self).__init__()

        # Input layer
        self.dropout = dropout_mapping[config.model.encoder.activation](
            config.model.encoder.regularization.input_dropout)
        self.fc = nn.Linear(config.model.encoder.input_dim,
                            config.model.encoder.number_hidden_neurons)
        self.act = activation_function_mapping[config.model.encoder.activation]

        # Hidden layer
        self.hidden_linear_layers = nn.ModuleList([])
        self.hidden_dropout_layers = nn.ModuleList([])
        self.hidden_activations = nn.ModuleList([])

        for _ in range(config.model.encoder.number_hidden_layers):
            self.hidden_dropout_layers.append(
                dropout_mapping[config.model.encoder.activation](
                    config.model.encoder.regularization.dropout)
            )
            self.hidden_linear_layers.append(
                nn.Linear(config.model.encoder.number_hidden_neurons,
                          config.model.encoder.number_hidden_neurons)
            )
            self.hidden_activations.append(
                activation_function_mapping[config.model.encoder.activation]
            )

        # Output layer
        self.dropout_o = dropout_mapping[config.model.encoder.activation](
            config.model.encoder.regularization.dropout)
        self.fc_o = nn.Linear(config.model.encoder.number_hidden_neurons,
                              config.model.associationSpace_dim)
        self.act_o = activation_function_mapping[config.model.encoder.activation]

        # Initialization
        encoder_initialization = partial(init_weights, config.model.encoder.activation)
        self.apply(encoder_initialization)

    def forward(self, molecule_representation):
        # Input layer
        x = self.dropout(molecule_representation)
        x = self.fc(x)
        x = self.act(x)

        # Hidden layer
        for hidden_dropout, hidden_layer, hidden_activation_function in zip(self.hidden_dropout_layers,
                                                                            self.hidden_linear_layers,
                                                                            self.hidden_activations):
            x = hidden_dropout(x)
            x = hidden_layer(x)
            x = hidden_activation_function(x)

        # Output layer
        x = self.dropout_o(x)
        x = self.fc_o(x)
        x = self.act_o(x)

        return x


class ContextModule(nn.Module):
    def __init__(self, config: OmegaConf):
        super(ContextModule, self).__init__()

        self.config = config

        self.hopfield = Hopfield(
            input_size=self.config.model.associationSpace_dim,
            hidden_size=config.model.hopfield.dim_QK,
            stored_pattern_size=self.config.model.associationSpace_dim,
            pattern_projection_size=self.config.model.associationSpace_dim,
            output_size=self.config.model.associationSpace_dim,
            num_heads=self.config.model.hopfield.heads,
            scaling=self.config.model.hopfield.beta,
            dropout=self.config.model.hopfield.dropout
        )

        # Initialization
        hopfield_initialization = partial(init_weights, 'linear')
        self.hopfield.apply(hopfield_initialization)

        self.padding_size = self.config.supportSet.stratified.paddingSize

    def forward(self, query_embedding, support_actives_embedding, support_inactives_embedding, reference_set_embedding):

        s = torch.cat((query_embedding, support_actives_embedding, support_inactives_embedding), dim=1)
        s_flattend = s.reshape(1, s.shape[0]*s.shape[1], s.shape[2])

        s_h = self.hopfield((reference_set_embedding, s_flattend, reference_set_embedding))

        s_updated = s_flattend + s_h
        s_updated_r = s_updated.reshape(s.shape[0], s.shape[1], s.shape[2])

        query_embedding = s_updated_r[:, 0, :]
        query_embedding = torch.unsqueeze(query_embedding, 1)

        support_actives_embedding = s_updated_r[:, 1:(self.padding_size + 1), :]
        support_inactives_embedding = s_updated_r[:, (self.padding_size + 1):, :]

        return query_embedding, support_actives_embedding, support_inactives_embedding


class LayerNormalizingBlock(nn.Module):
    def __init__(self, config: OmegaConf):
        super(LayerNormalizingBlock, self).__init__()

        self.config = config

        if config.model.layerNormBlock.usage:
            self.layernorm_query = nn.LayerNorm(config.model.associationSpace_dim,
                                                elementwise_affine=config.model.layerNormBlock.affine)
            self.layernorm_support_actives = nn.LayerNorm(config.model.associationSpace_dim,
                                                          elementwise_affine=config.model.layerNormBlock.affine)
            self.layernorm_support_inactives = nn.LayerNorm(config.model.associationSpace_dim,
                                                            elementwise_affine=config.model.layerNormBlock.affine)

    def forward(self, query_embedding, support_actives_embedding, support_inactives_embedding):
        # Layer normalization
        if self.config.model.layerNormBlock.usage:
            query_embedding = self.layernorm_query(query_embedding)
            support_actives_embedding = self.layernorm_support_actives(support_actives_embedding)
            if support_inactives_embedding is not None:
                support_inactives_embedding = self.layernorm_support_Inactives(support_inactives_embedding)
        return query_embedding, support_actives_embedding, support_inactives_embedding


class CrossAttentionModule(nn.Module):
    def __init__(self, config: OmegaConf):
        super(CrossAttentionModule, self).__init__()

        self.config = config

        transformer_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=(self.config.model.associationSpace_dim + self.config.model.transformer.activity_embedding_dim),
            nhead=self.config.model.transformer.number_heads,
            dim_feedforward=self.config.model.transformer.dim_forward,
            dropout=self.config.model.transformer.dropout
        )
        self.transformer = torch.nn.TransformerEncoder(transformer_encoder_layer,
                                                       num_layers=self.config.model.transformer.num_layers)

        self.padding_size = self.config.supportSet.stratified.paddingSize

    def forward(self, query_embedding, support_actives_embedding, support_inactives_embedding,
                support_set_actives_size, support_set_inactives_size):

        embedding_dim = support_actives_embedding.shape[2]
        query_embedding = torch.cat([query_embedding, torch.zeros_like(query_embedding[:, :, :64])], dim=2)
        support_actives_embedding = torch.cat([support_actives_embedding,
                                               torch.ones_like(support_actives_embedding[:, :, :64])], dim=2)
        support_inactives_embedding = torch.cat([support_inactives_embedding,
                                                (-1.) * torch.ones_like(support_inactives_embedding[:, :, :64])], dim=2)

        s = torch.cat([query_embedding, support_actives_embedding, support_inactives_embedding], dim=1)
        src_key_padding_mask = torch.zeros(s.shape[0], s.shape[1]).to(s.device)

        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                real_actives = support_set_actives_size[i]
                real_inactives = support_set_inactives_size[i]

                if j == 0:
                    src_key_padding_mask[i, j] = 1
                elif j < (self.padding_size + 1):
                    if j < (real_actives + 1):
                        src_key_padding_mask[i, j] = 1
                else:
                    if j < (real_inactives + (self.padding_size + 1)):
                        src_key_padding_mask[i, j] = 1
        src_key_padding_mask = src_key_padding_mask.bool()

        s = torch.transpose(s, 0, 1)

        s_h = self.transformer(s, src_key_padding_mask=src_key_padding_mask)
        s = torch.transpose(s, 0, 1)
        s_h = torch.transpose(s_h, 0, 1)
        s_updated = s + s_h

        query_embedding = s_updated[:, 0, :embedding_dim]
        query_embedding = torch.unsqueeze(query_embedding, 1)
        support_actives_embedding = s_updated[:, 1:(self.padding_size + 1), :embedding_dim]
        support_inactives_embedding = s_updated[:, (self.padding_size + 1):, :embedding_dim]

        return query_embedding, support_actives_embedding, support_inactives_embedding


def similarity_module(query_embedding, support_set_embeddings, support_set_size, config):

    # L2-Norm
    if config.model.similarityModule.l2Norm:
        query_embedding_div = torch.unsqueeze(query_embedding.pow(2).sum(dim=2).sqrt(), 2)
        query_embedding_div[query_embedding_div == 0] = 1
        support_set_embeddings_div = torch.unsqueeze(support_set_embeddings.pow(2).sum(dim=2).sqrt(), 2)
        support_set_embeddings_div[support_set_embeddings_div == 0] = 1

        query_embedding = query_embedding / query_embedding_div
        support_set_embeddings = support_set_embeddings / support_set_embeddings_div

    similarities = query_embedding @ torch.transpose(support_set_embeddings, 1, 2)

    # Masking: Remove padded support set artefacts
    mask = torch.zeros_like(similarities)
    for task_idx in range(support_set_embeddings.shape[0]):
        real_size = support_set_size[task_idx]
        if real_size > 0:
            mask[task_idx, :, :real_size] = torch.ones_like(mask[task_idx, :, :real_size])

    # Compute similarity values
    similarities = similarities * mask
    similarity_sums = similarities.sum(dim=2)  # For every query molecule: Sum over support set molecules

    # Scaling
    if config.model.similarityModule.scaling == '1/N':
        stabilizer = torch.tensor(1e-8).float()
        similarity_sums = 1/(2.*support_set_size.reshape(-1, 1) + stabilizer) * similarity_sums
    if config.model.similarityModule.scaling == '1/sqrt(N)':
        stabilizer = torch.tensor(1e-8).float()
        similarity_sums = 1 / (2.*torch.sqrt(support_set_size.reshape(-1, 1).float()) + stabilizer) * similarity_sums

    return similarity_sums
