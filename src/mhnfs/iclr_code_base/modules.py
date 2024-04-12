import torch.nn as nn
import torch
import numpy as np
import copy
from functools import partial
from omegaconf import OmegaConf
import sys
sys.path.append('.')
from src.mhnfs.hopfield.modules import Hopfield
from src.mhnfs.initialization import init_weights

# Mappings
activation_function_mapping = {
    "relu": nn.ReLU(),
    "selu": nn.SELU(),
    "sigmoid": nn.Sigmoid(),
}

dropout_mapping = {"relu": nn.Dropout, "selu": nn.AlphaDropout}


# Modules


class EncoderBlock(nn.Module):
    def __init__(self, config: OmegaConf):
        super(EncoderBlock, self).__init__()

        # Input layer
        self.dropout = dropout_mapping[config.model.encoder.activation](
            config.model.encoder.regularization.input_dropout
        )
        self.fc = nn.Linear(
            config.model.encoder.input_dim, config.model.encoder.number_hidden_neurons
        )
        self.act = activation_function_mapping[config.model.encoder.activation]

        # Hidden layer
        self.hidden_linear_layers = nn.ModuleList([])
        self.hidden_dropout_layers = nn.ModuleList([])
        self.hidden_activations = nn.ModuleList([])

        for _ in range(config.model.encoder.number_hidden_layers):
            self.hidden_dropout_layers.append(
                dropout_mapping[config.model.encoder.activation](
                    config.model.encoder.regularization.dropout
                )
            )
            self.hidden_linear_layers.append(
                nn.Linear(
                    config.model.encoder.number_hidden_neurons,
                    config.model.encoder.number_hidden_neurons,
                )
            )
            self.hidden_activations.append(
                activation_function_mapping[config.model.encoder.activation]
            )

        # Output layer
        self.dropout_o = dropout_mapping[config.model.encoder.activation](
            config.model.encoder.regularization.dropout
        )
        self.fc_o = nn.Linear(
            config.model.encoder.number_hidden_neurons,
            config.model.associationSpace_dim,
        )
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
        for hidden_dropout, hidden_layer, hidden_activation_function in zip(
            self.hidden_dropout_layers,
            self.hidden_linear_layers,
            self.hidden_activations,
        ):
            x = hidden_dropout(x)
            x = hidden_layer(x)
            x = hidden_activation_function(x)

        # Output layer
        x = self.dropout_o(x)
        x = self.fc_o(x)
        x = self.act_o(x)

        return x


class LayerNormalizingBlock(nn.Module):
    def __init__(self, config: OmegaConf):
        super(LayerNormalizingBlock, self).__init__()

        self.config = config

        if config.model.layerNormBlock.usage == True:
            self.layernorm_query = nn.LayerNorm(
                config.model.associationSpace_dim,
                elementwise_affine=config.model.layerNormBlock.affine,
            )
            self.layernorm_support_Actives = nn.LayerNorm(
                config.model.associationSpace_dim,
                elementwise_affine=config.model.layerNormBlock.affine,
            )
            self.layernorm_support_Inactives = nn.LayerNorm(
                config.model.associationSpace_dim,
                elementwise_affine=config.model.layerNormBlock.affine,
            )

    def forward(
        self, query_embedding, supportActives_embedding, supportInactives_embedding
    ):
        # Layer normalization
        if self.config.model.layerNormBlock.usage == True:
            query_embedding = self.layernorm_query(query_embedding)
            supportActives_embedding = self.layernorm_support_Actives(
                supportActives_embedding
            )
            if supportInactives_embedding != None:
                supportInactives_embedding = self.layernorm_support_Inactives(
                    supportInactives_embedding
                )
        return query_embedding, supportActives_embedding, supportInactives_embedding


class LayerNormalizingBlockSepp(nn.Module):
    def __init__(self, config: OmegaConf):
        super(LayerNormalizingBlockSepp, self).__init__()

        self.config = config

        if config.model.layerNormBlock.usage == True:
            self.layernorm_query = nn.LayerNorm(
                config.model.associationSpace_dim,
                elementwise_affine=config.model.layerNormBlock.affine,
            )
            self.layernorm_support_Actives = nn.LayerNorm(
                config.model.associationSpace_dim,
                elementwise_affine=config.model.layerNormBlock.affine,
            )
            self.layernorm_support_Inactives = nn.LayerNorm(
                config.model.associationSpace_dim,
                elementwise_affine=config.model.layerNormBlock.affine,
            )
            self.layernorm_memory_TrainMols = nn.LayerNorm(
                config.model.associationSpace_dim,
                elementwise_affine=config.model.layerNormBlock.affine,
            )

    def forward(
        self,
        query_embedding,
        supportActives_embedding,
        supportInactives_embedding,
        memory_TrainMols,
    ):
        # Layer normalization
        if self.config.model.layerNormBlock.usage == True:
            query_embedding = self.layernorm_query(query_embedding)
            supportActives_embedding = self.layernorm_support_Actives(
                supportActives_embedding
            )
            if supportInactives_embedding != None:
                supportInactives_embedding = self.layernorm_support_Inactives(
                    supportInactives_embedding
                )
            memory_TrainMols_embedding = self.layernorm_memory_TrainMols(
                memory_TrainMols
            )

        return (
            query_embedding,
            supportActives_embedding,
            supportInactives_embedding,
            memory_TrainMols_embedding,
        )


class HopfieldBlock_old(nn.Module):
    def __init__(self, config: OmegaConf):
        super(HopfieldBlock, self).__init__()

        self.hopfield = Hopfield(
            input_size=562,  # config.model.encoder.number_hidden_neurons,      # R
            hidden_size=config.model.hopfield.dim_QK,  # a_1, Dimension Queries, Keys
            stored_pattern_size=562,  # config.model.encoder.number_hidden_neurons,           # Y
            pattern_projection_size=512,  # config.model.encoder.number_hidden_neurons,       # Y
            output_size=config.model.associationSpace_dim,  # a_2, Dim Values / Dim Dotproduct
            num_heads=config.model.hopfield.heads,
            scaling=config.model.hopfield.beta,
        )

        # Hopfield 2
        # self.hopfield2 = Hopfield(
        #    input_size=512,  # config.model.encoder.number_hidden_neurons,      # R
        #    hidden_size=config.model.hopfield.dim_QK,  # a_1, Dimension Queries, Keys
        #    stored_pattern_size=512,  # config.model.encoder.number_hidden_neurons,           # Y
        #    pattern_projection_size=512,  # config.model.encoder.number_hidden_neurons,       # Y
        #    output_size=config.model.associationSpace_dim,  # a_2, Dim Values / Dim Dotproduct
        #    num_heads=config.model.hopfield.heads,
        #    scaling=config.model.hopfield.beta
        # )

        # Hopfield 3
        # self.hopfield3 = Hopfield(
        #    input_size=512,  # config.model.encoder.number_hidden_neurons,      # R
        #    hidden_size=config.model.hopfield.dim_QK,  # a_1, Dimension Queries, Keys
        #    stored_pattern_size=512,  # config.model.encoder.number_hidden_neurons,           # Y
        #    pattern_projection_size=512,  # config.model.encoder.number_hidden_neurons,       # Y
        #    output_size=config.model.associationSpace_dim,  # a_2, Dim Values / Dim Dotproduct
        #   num_heads=config.model.hopfield.heads,
        #    scaling=config.model.hopfield.beta
        # )

        # Initialization
        hopfield_initialization = partial(init_weights, "linear")
        self.hopfield.apply(hopfield_initialization)

    # def forward(self,R, Y):
    def forward(self, SupportSetActivesEmbedding, SupportSetInactivesEmbedding):
        # Unsqueeze R, Y
        # R = torch.unsqueeze(R, dim=1)

        # Y = torch.unsqueeze(Y, dim=0)

        # Hopfield operation

        # temp
        # SupportSetActivesEmbedding = self.hopfield((SupportSetActivesEmbedding, SupportSetActivesEmbedding,
        #                                            SupportSetActivesEmbedding))
        if SupportSetInactivesEmbedding == "None":
            SupportSetActivesEmbedding = self.hopfield(
                (
                    SupportSetActivesEmbedding,
                    SupportSetActivesEmbedding,
                    SupportSetActivesEmbedding,
                )
            )  # TODO give mask
        else:
            numb_actives = SupportSetActivesEmbedding.shape[
                1
            ]  # ToDo flexibel for ratio
            numb_describtors = SupportSetActivesEmbedding.shape[2]
            SupportSetActivesEmbedding = torch.cat(
                [
                    SupportSetActivesEmbedding,
                    torch.ones_like(SupportSetActivesEmbedding[:, :, :50]),
                ],
                dim=2,
            )
            SupportSetInactivesEmbedding = torch.cat(
                [
                    SupportSetInactivesEmbedding,
                    torch.zeros_like(SupportSetInactivesEmbedding[:, :, :50]),
                ],
                dim=2,
            )

            S_init = torch.cat(
                [SupportSetActivesEmbedding, SupportSetInactivesEmbedding], axis=1
            )
            S_ValuesInput = S_init[:, :, :numb_describtors]
            S = self.hopfield((S_init, S_init, S_ValuesInput))
            # S3 = self.hopfield2((S2, S2, S2))
            # S3 = S2
            # S = self.hopfield3((S3,S3, S_ValuesInput))

            # Todo: second hopfield layer
            # First hopfield: (S,S,S)
            # Second (S,S,S*)

            SupportSetActivesEmbedding = S[:, :numb_actives, :]
            SupportSetInactivesEmbedding = S[:, numb_actives:, :]

        return SupportSetActivesEmbedding, SupportSetInactivesEmbedding


class HopfieldBlock(nn.Module):
    def __init__(self, config: OmegaConf):
        super(HopfieldBlock, self).__init__()

        self.config = config

        self.hopfield = Hopfield(
            input_size=self.config.model.associationSpace_dim,  # R
            hidden_size=config.model.hopfield.dim_QK,  # a_1, Dimension Queries, Keys
            stored_pattern_size=self.config.model.associationSpace_dim,  # Y
            pattern_projection_size=self.config.model.associationSpace_dim,  # Y
            output_size=self.config.model.associationSpace_dim,  # a_2, Dim Values / Dim Dotproduct
            num_heads=self.config.model.hopfield.heads,
            scaling=self.config.model.hopfield.beta,
            dropout=self.config.model.hopfield.dropout,
        )

        # Initialization
        hopfield_initialization = partial(init_weights, "linear")
        self.hopfield.apply(hopfield_initialization)

    def forward(
        self,
        query_embedding,
        supportActives_embedding,
        supportInactives_embedding,
        supportSetActivesSize,
        supportSetInactivesSize,
    ):
        embedding_dim = supportActives_embedding.shape[2]
        query_embedding = torch.cat(
            [query_embedding, torch.zeros_like(query_embedding[:, :, :64])], dim=2
        )
        supportActives_embedding = torch.cat(
            [
                supportActives_embedding,
                torch.ones_like(supportActives_embedding[:, :, :64]),
            ],
            dim=2,
        )
        supportInactives_embedding = torch.cat(
            [
                supportInactives_embedding,
                (-1.0) * torch.ones_like(supportInactives_embedding[:, :, :64]),
            ],
            dim=2,
        )

        S = torch.cat(
            [query_embedding, supportActives_embedding, supportInactives_embedding],
            axis=1,
        )
        src_key_padding_mask = torch.zeros(S.shape[0], S.shape[1]).to(S.device)

        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                real_actives = supportSetActivesSize[i]
                real_inactives = supportSetInactivesSize[i]

                if j == 0:
                    src_key_padding_mask[i, j] = 1
                elif j < 13:
                    if j < (real_actives + 1):
                        src_key_padding_mask[i, j] = 1
                else:
                    if j < (real_inactives + 1 + 12):
                        src_key_padding_mask[i, j] = 1
        src_key_padding_mask = src_key_padding_mask.bool()

        # S = torch.transpose(S,0,1)

        S_h = self.hopfield((S, S, S), stored_pattern_padding_mask=src_key_padding_mask)

        S_updated = S + S_h
        # S_updated = torch.transpose(S_updated, 0, 1)

        query_embedding = S_updated[:, 0, :embedding_dim]
        query_embedding = torch.unsqueeze(query_embedding, 1)
        supportActives_embedding = S_updated[:, 1:13, :embedding_dim]
        supportInactives_embedding = S_updated[:, 13:, :embedding_dim]

        return query_embedding, supportActives_embedding, supportInactives_embedding


class IterRefEmbedding(nn.Module):
    def __init__(self, config: OmegaConf):
        super(IterRefEmbedding, self).__init__()

        self.config = config

        self.lstm_s = torch.nn.LSTM(
            input_size=config.model.associationSpace_dim * 2,
            hidden_size=config.model.associationSpace_dim,
            batch_first=True,
        )

        self.lstm_q = torch.nn.LSTM(
            input_size=config.model.associationSpace_dim * 2,
            hidden_size=config.model.associationSpace_dim,
            batch_first=True,
        )

    def forward(
        self,
        query_representation,
        support_set_actives_representation,
        support_set_inactives_representation,
    ):
        def cosine_distance(x, y):
            div_stabilizer = torch.tensor([1e-8]).to(x.device)

            x_norm = x.norm(p=2, dim=2, keepdim=True)
            x = x.div(x_norm.expand_as(x) + div_stabilizer)

            y_norm = y.norm(p=2, dim=2, keepdim=True)
            y = y.div(y_norm.expand_as(y) + div_stabilizer)

            sim = x @ torch.transpose(y, 1, 2)

            return sim

        # Initialization:
        # Initialize refinement delta values
        support_set_representation = torch.cat(
            [support_set_actives_representation, support_set_inactives_representation],
            1,
        )
        q_refine = torch.zeros_like(query_representation)
        s_refine = torch.zeros_like(support_set_representation)

        # Initialize temp set of attention mechanism
        z = support_set_representation

        # Initialize states for lstms
        h_s = torch.unsqueeze(
            torch.zeros_like(
                support_set_representation.reshape(
                    -1, self.config.model.associationSpace_dim
                )
            ),
            0,
        )
        c_s = torch.unsqueeze(
            torch.zeros_like(
                support_set_representation.reshape(
                    -1, self.config.model.associationSpace_dim
                )
            ),
            0,
        )
        h_q = torch.unsqueeze(
            torch.zeros_like(
                query_representation.reshape(-1, self.config.model.associationSpace_dim)
            ),
            0,
        )
        c_q = torch.unsqueeze(
            torch.zeros_like(
                query_representation.reshape(-1, self.config.model.associationSpace_dim)
            ),
            0,
        )

        for i in range(self.config.model.number_iteration_steps):
            # Attention mechanism
            # - Support set
            cosine_sim_s = cosine_distance(z + s_refine, support_set_representation)
            attention_values_s = torch.nn.Softmax(dim=2)(cosine_sim_s)
            linear_comb_s = attention_values_s @ support_set_representation

            # - Query
            cosine_sim_q = cosine_distance(query_representation + q_refine, z)
            attention_values_q = torch.nn.Softmax(dim=2)(cosine_sim_q)
            linear_comb_q = attention_values_q @ z

            # Concatenate and prepare variables for lstms
            s_lstm_in = torch.cat([s_refine, linear_comb_s], dim=2)
            q_lstm_in = torch.cat([q_refine, linear_comb_q], dim=2)

            # Feed inputs in lstm
            s_lstm_in = torch.unsqueeze(
                s_lstm_in.reshape(-1, self.config.model.associationSpace_dim * 2), 1
            )
            s_refine, (h_s, c_s) = self.lstm_s(s_lstm_in, (h_s, c_s))
            s_refine = s_refine.reshape(-1, 24, self.config.model.associationSpace_dim)

            q_lstm_in = torch.unsqueeze(
                q_lstm_in.reshape(-1, self.config.model.associationSpace_dim * 2), 1
            )
            q_refine, (h_q, c_q) = self.lstm_q(q_lstm_in, (h_q, c_q))
            q_refine = q_refine.reshape(-1, 1, self.config.model.associationSpace_dim)

            # Update temp set for attention mechnism
            z = linear_comb_s

        q_updated = query_representation + q_refine
        s_updated = support_set_representation + s_refine

        s_updated_actices = s_updated[:, :12, :]
        s_updated_inactices = s_updated[:, 12:, :]

        return q_updated, s_updated_actices, s_updated_inactices


class IterRefEmbedding_vartest(nn.Module):
    def __init__(self, config: OmegaConf):
        super(IterRefEmbedding_vartest, self).__init__()

        self.config = config

        self.lstm_s = torch.nn.LSTM(
            input_size=config.model.associationSpace_dim * 2,
            hidden_size=config.model.associationSpace_dim,
            batch_first=True,
        )

        self.lstm_q = torch.nn.LSTM(
            input_size=config.model.associationSpace_dim * 2,
            hidden_size=config.model.associationSpace_dim,
            batch_first=True,
        )

    def forward(
        self,
        query_representation,
        support_set_actives_representation,
        support_set_inactives_representation,
    ):
        def cosine_distance(x, y):
            div_stabilizer = torch.tensor([1e-8]).to(x.device)

            x_norm = x.norm(p=2, dim=2, keepdim=True)
            x = x.div(x_norm.expand_as(x) + div_stabilizer)

            y_norm = y.norm(p=2, dim=2, keepdim=True)
            y = y.div(y_norm.expand_as(y) + div_stabilizer)

            sim = x @ torch.transpose(y, 1, 2)

            return sim

        # Initialization:
        # Initialize refinement delta values
        support_set_representation = torch.cat(
            [support_set_actives_representation, support_set_inactives_representation],
            1,
        )
        q_refine = torch.zeros_like(query_representation)
        s_refine = torch.zeros_like(support_set_representation)

        # Initialize temp set of attention mechanism
        z = support_set_representation

        # Initialize states for lstms
        h_s = torch.unsqueeze(
            torch.zeros_like(
                support_set_representation.reshape(
                    -1, self.config.model.associationSpace_dim
                )
            ),
            0,
        )
        c_s = torch.unsqueeze(
            torch.zeros_like(
                support_set_representation.reshape(
                    -1, self.config.model.associationSpace_dim
                )
            ),
            0,
        )
        h_q = torch.unsqueeze(
            torch.zeros_like(
                query_representation.reshape(-1, self.config.model.associationSpace_dim)
            ),
            0,
        )
        c_q = torch.unsqueeze(
            torch.zeros_like(
                query_representation.reshape(-1, self.config.model.associationSpace_dim)
            ),
            0,
        )

        for i in range(self.config.model.number_iteration_steps):
            # Attention mechanism
            # - Support set
            cosine_sim_s = cosine_distance(z + s_refine, support_set_representation)
            attention_values_s = torch.nn.Softmax(dim=2)(cosine_sim_s)
            linear_comb_s = attention_values_s @ support_set_representation

            # - Query
            cosine_sim_q = cosine_distance(query_representation + q_refine, z)
            attention_values_q = torch.nn.Softmax(dim=2)(cosine_sim_q)
            linear_comb_q = attention_values_q @ z

            # Concatenate and prepare variables for lstms
            s_lstm_in = torch.cat([s_refine, linear_comb_s], dim=2)
            q_lstm_in = torch.cat([q_refine, linear_comb_q], dim=2)

            # Feed inputs in lstm
            s_lstm_in = torch.unsqueeze(
                s_lstm_in.reshape(-1, self.config.model.associationSpace_dim * 2), 1
            )
            s_refine, (h_s, c_s) = self.lstm_s(s_lstm_in, (h_s, c_s))
            s_refine = s_refine.reshape(-1, 256, self.config.model.associationSpace_dim)

            q_lstm_in = torch.unsqueeze(
                q_lstm_in.reshape(-1, self.config.model.associationSpace_dim * 2), 1
            )
            q_refine, (h_q, c_q) = self.lstm_q(q_lstm_in, (h_q, c_q))
            q_refine = q_refine.reshape(-1, 1, self.config.model.associationSpace_dim)

            # Update temp set for attention mechnism
            z = linear_comb_s

        q_updated = query_representation + q_refine
        s_updated = support_set_representation + s_refine

        s_updated_actices = s_updated[:, :128, :]
        s_updated_inactices = s_updated[:, 128:, :]

        return q_updated, s_updated_actices, s_updated_inactices


class TransformerEmbedding(nn.Module):
    def __init__(self, config: OmegaConf):
        super(TransformerEmbedding, self).__init__()

        self.config = config

        transformerEncoderLayer = torch.nn.TransformerEncoderLayer(
            d_model=(
                self.config.model.associationSpace_dim
                + self.config.model.transformer.activity_embedding_dim
            ),
            nhead=self.config.model.transformer.number_heads,
            dim_feedforward=self.config.model.transformer.dim_forward,
            dropout=self.config.model.transformer.dropout,
        )
        self.transformer = torch.nn.TransformerEncoder(
            transformerEncoderLayer, num_layers=self.config.model.transformer.num_layers
        )

    def forward(
        self,
        query_embedding,
        supportActives_embedding,
        supportInactives_embedding,
        supportSetActivesSize,
        supportSetInactivesSize,
    ):
        embedding_dim = supportActives_embedding.shape[2]
        query_embedding = torch.cat(
            [query_embedding, torch.zeros_like(query_embedding[:, :, :64])], dim=2
        )
        supportActives_embedding = torch.cat(
            [
                supportActives_embedding,
                torch.ones_like(supportActives_embedding[:, :, :64]),
            ],
            dim=2,
        )
        supportInactives_embedding = torch.cat(
            [
                supportInactives_embedding,
                (-1.0) * torch.ones_like(supportInactives_embedding[:, :, :64]),
            ],
            dim=2,
        )

        S = torch.cat(
            [query_embedding, supportActives_embedding, supportInactives_embedding],
            axis=1,
        )
        src_key_padding_mask = torch.zeros(S.shape[0], S.shape[1]).to(S.device)

        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                real_actives = supportSetActivesSize[i]
                real_inactives = supportSetInactivesSize[i]

                if j == 0:
                    src_key_padding_mask[i, j] = 1
                elif j < 13:
                    if j < (real_actives + 1):
                        src_key_padding_mask[i, j] = 1
                else:
                    if j < (real_inactives + 1 + 12):
                        src_key_padding_mask[i, j] = 1
        src_key_padding_mask = ~src_key_padding_mask.bool()  # TODO add ~

        S = torch.transpose(S, 0, 1)

        S_h = self.transformer(S, src_key_padding_mask=src_key_padding_mask)
        S = torch.transpose(S, 0, 1)
        S_h = torch.transpose(S_h, 0, 1)
        s_updated = S + S_h

        query_embedding = s_updated[:, 0, :embedding_dim]
        query_embedding = torch.unsqueeze(query_embedding, 1)
        supportActives_embedding = s_updated[:, 1:13, :embedding_dim]
        supportInactives_embedding = s_updated[:, 13:, :embedding_dim]

        return query_embedding, supportActives_embedding, supportInactives_embedding


class TransformerEmbedding_testvar(nn.Module):
    def __init__(self, config: OmegaConf):
        super(TransformerEmbedding_testvar, self).__init__()

        self.config = config

        transformerEncoderLayer = torch.nn.TransformerEncoderLayer(
            d_model=(
                self.config.model.associationSpace_dim
                + self.config.model.transformer.activity_embedding_dim
            ),
            nhead=self.config.model.transformer.number_heads,
            dim_feedforward=self.config.model.transformer.dim_forward,
            dropout=self.config.model.transformer.dropout,
        )
        self.transformer = torch.nn.TransformerEncoder(
            transformerEncoderLayer, num_layers=self.config.model.transformer.num_layers
        )

    def forward(
        self,
        query_embedding,
        supportActives_embedding,
        supportInactives_embedding,
        supportSetActivesSize,
        supportSetInactivesSize,
    ):
        embedding_dim = supportActives_embedding.shape[2]
        query_embedding = torch.cat(
            [query_embedding, torch.zeros_like(query_embedding[:, :, :64])], dim=2
        )
        supportActives_embedding = torch.cat(
            [
                supportActives_embedding,
                torch.ones_like(supportActives_embedding[:, :, :64]),
            ],
            dim=2,
        )
        supportInactives_embedding = torch.cat(
            [
                supportInactives_embedding,
                (-1.0) * torch.ones_like(supportInactives_embedding[:, :, :64]),
            ],
            dim=2,
        )

        S = torch.cat(
            [query_embedding, supportActives_embedding, supportInactives_embedding],
            axis=1,
        )
        src_key_padding_mask = torch.zeros(S.shape[0], S.shape[1]).to(S.device)

        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                real_actives = supportSetActivesSize[i]
                real_inactives = supportSetInactivesSize[i]

                if j == 0:
                    src_key_padding_mask[i, j] = 1
                elif j < 129:
                    if j < (real_actives + 1):
                        src_key_padding_mask[i, j] = 1
                else:
                    if j < (real_inactives + 1 + 128):
                        src_key_padding_mask[i, j] = 1
        src_key_padding_mask = src_key_padding_mask.bool()

        S = torch.transpose(S, 0, 1)

        S_h = self.transformer(S, src_key_padding_mask=src_key_padding_mask)
        S = torch.transpose(S, 0, 1)
        S_h = torch.transpose(S_h, 0, 1)
        s_updated = S + S_h

        query_embedding = s_updated[:, 0, :embedding_dim]
        query_embedding = torch.unsqueeze(query_embedding, 1)
        supportActives_embedding = s_updated[:, 1:129, :embedding_dim]
        supportInactives_embedding = s_updated[:, 129:, :embedding_dim]

        return query_embedding, supportActives_embedding, supportInactives_embedding


class TransformerEmbedding_referenceSet(nn.Module):
    def __init__(self, config: OmegaConf):
        super(TransformerEmbedding_referenceSet, self).__init__()

        self.config = config

        transformerEncoderLayer = torch.nn.TransformerEncoderLayer(
            d_model=(
                self.config.model.associationSpace_dim
                + self.config.model.transformer.activity_embedding_dim
            ),
            nhead=self.config.model.transformer.number_heads,
            dim_feedforward=self.config.model.transformer.dim_forward,
            dropout=self.config.model.transformer.dropout,
        )
        self.transformer = torch.nn.TransformerEncoder(
            transformerEncoderLayer, num_layers=self.config.model.transformer.num_layers
        )

    def forward(
        self,
        query_embedding,
        supportActives_embedding,
        supportInactives_embedding,
        supportSetActivesSize,
        supportSetInactivesSize,
        referenceSet,
    ):
        embedding_dim = supportActives_embedding.shape[2]
        query_embedding = torch.cat(
            [query_embedding, torch.zeros_like(query_embedding[:, :, :64])], dim=2
        )
        referenceSet = torch.cat(
            [referenceSet, torch.zeros_like(referenceSet[:, :, :64])], dim=2
        )
        supportActives_embedding = torch.cat(
            [
                supportActives_embedding,
                torch.ones_like(supportActives_embedding[:, :, :64]),
            ],
            dim=2,
        )
        supportInactives_embedding = torch.cat(
            [
                supportInactives_embedding,
                (-1.0) * torch.ones_like(supportInactives_embedding[:, :, :64]),
            ],
            dim=2,
        )

        S = torch.cat(
            [
                query_embedding,
                supportActives_embedding,
                supportInactives_embedding,
                referenceSet,
            ],
            axis=1,
        )
        src_key_padding_mask = torch.zeros(S.shape[0], S.shape[1]).to(S.device)

        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                real_actives = supportSetActivesSize[i]
                real_inactives = supportSetInactivesSize[i]

                if j == 0:
                    src_key_padding_mask[i, j] = 1
                elif j < 13:
                    if j < (real_actives + 1):
                        src_key_padding_mask[i, j] = 1
                else:
                    if j < (real_inactives + 1 + 12):
                        src_key_padding_mask[i, j] = 1
        src_key_padding_mask = src_key_padding_mask.bool()

        S = torch.transpose(S, 0, 1)

        S = self.transformer(S, src_key_padding_mask=src_key_padding_mask)
        S = torch.transpose(S, 0, 1)

        query_embedding = S[:, 0, :embedding_dim]
        query_embedding = torch.unsqueeze(query_embedding, 1)
        supportActives_embedding = S[:, 1:13, :embedding_dim]
        supportInactives_embedding = S[:, 13:25, :embedding_dim]

        return query_embedding, supportActives_embedding, supportInactives_embedding


class HopfieldBlock_chemTrainSpace(nn.Module):
    def __init__(self, config: OmegaConf):
        super(HopfieldBlock_chemTrainSpace, self).__init__()

        self.config = config

        self.hopfield = Hopfield(
            input_size=self.config.model.associationSpace_dim,  # R
            hidden_size=config.model.hopfield.dim_QK,  # a_1, Dimension Queries, Keys
            stored_pattern_size=self.config.model.associationSpace_dim,  # Y
            pattern_projection_size=self.config.model.associationSpace_dim,  # Y
            output_size=self.config.model.associationSpace_dim,  # a_2, Dim Values / Dim Dotproduct
            num_heads=self.config.model.hopfield.heads,
            scaling=self.config.model.hopfield.beta,
            dropout=self.config.model.hopfield.dropout,
        )

        # self.hopfield = Hopfield(
        #    input_size=self.config.model.associationSpace_dim, # R
        #    #hidden_size=config.model.hopfield.dim_QK,                   # a_1, Dimension Queries, Keys
        #    #stored_pattern_size=self.config.model.associationSpace_dim, # Y
        #    #pattern_projection_size=self.config.model.associationSpace_dim,# Y
        #    output_size=self.config.model.associationSpace_dim, # a_2, Dim Values / Dim Dotproduct
        #    #num_heads=self.config.model.hopfield.heads,
        #    scaling=self.config.model.hopfield.beta,
        #    dropout=self.config.model.hopfield.dropout,
        #    stored_pattern_as_static=True,  # type: bool
        #    state_pattern_as_static=True,  # type: bool
        #    pattern_projection_as_static=True
        # )

        # Initialization
        hopfield_initialization = partial(init_weights, "linear")
        self.hopfield.apply(hopfield_initialization)

    def forward(
        self,
        query_embedding,
        supportActives_embedding,
        supportInactives_embedding,
        supportSetActivesSize,
        supportSetInactivesSize,
        referenceSet_embedding,
    ):
        S = torch.cat(
            [query_embedding, supportActives_embedding, supportInactives_embedding],
            dim=1,
        )

        S_flattend = S.reshape(1, S.shape[0] * S.shape[1], S.shape[2])

        S_h = self.hopfield(
            (referenceSet_embedding, S_flattend, referenceSet_embedding)
        )

        S_updated = S_flattend + S_h
        S_updated_r = S_updated.reshape(S.shape[0], S.shape[1], S.shape[2])

        query_embedding = S_updated_r[:, 0, :]
        query_embedding = torch.unsqueeze(query_embedding, 1)
        supportActives_embedding = S_updated_r[:, 1:13, :]
        supportInactives_embedding = S_updated_r[:, 13:, :]

        return query_embedding, supportActives_embedding, supportInactives_embedding


class HopfieldBlock_chemTrainSpace_testvar(nn.Module):
    def __init__(self, config: OmegaConf):
        super(HopfieldBlock_chemTrainSpace_testvar, self).__init__()

        self.config = config

        self.hopfield = Hopfield(
            input_size=self.config.model.associationSpace_dim,  # R
            hidden_size=config.model.hopfield.dim_QK,  # a_1, Dimension Queries, Keys
            stored_pattern_size=self.config.model.associationSpace_dim,  # Y
            pattern_projection_size=self.config.model.associationSpace_dim,  # Y
            output_size=self.config.model.associationSpace_dim,  # a_2, Dim Values / Dim Dotproduct
            num_heads=self.config.model.hopfield.heads,
            scaling=self.config.model.hopfield.beta,
            dropout=self.config.model.hopfield.dropout,
        )

        # Initialization
        hopfield_initialization = partial(init_weights, "linear")
        self.hopfield.apply(hopfield_initialization)

    def forward(
        self,
        query_embedding,
        supportActives_embedding,
        supportInactives_embedding,
        supportSetActivesSize,
        supportSetInactivesSize,
        referenceSet_embedding,
    ):
        S = torch.cat(
            [query_embedding, supportActives_embedding, supportInactives_embedding],
            axis=1,
        )

        S_flattend = S.reshape(1, S.shape[0] * S.shape[1], S.shape[2])

        S_h = self.hopfield(
            (referenceSet_embedding, S_flattend, referenceSet_embedding)
        )

        S_updated = S_flattend + S_h
        S_updated_r = S_updated.reshape(S.shape[0], S.shape[1], S.shape[2])

        query_embedding = S_updated_r[:, 0, :]
        query_embedding = torch.unsqueeze(query_embedding, 1)
        supportActives_embedding = S_updated_r[:, 1:129, :]
        supportInactives_embedding = S_updated_r[:, 129:, :]

        return query_embedding, supportActives_embedding, supportInactives_embedding


class HopfieldBlock_chemTrainSpaceActivity(nn.Module):
    def __init__(self, config: OmegaConf):
        super(HopfieldBlock_chemTrainSpaceActivity, self).__init__()

        self.config = config

        self.hopfield = Hopfield(
            input_size=self.config.model.associationSpace_dim,  # R
            hidden_size=config.model.hopfield.dim_QK,  # a_1, Dimension Queries, Keys
            stored_pattern_size=self.config.model.associationSpace_dim + 4938,  # Y
            pattern_projection_size=self.config.model.associationSpace_dim + 4938,  # Y
            output_size=self.config.model.associationSpace_dim,  # a_2, Dim Values / Dim Dotproduct
            num_heads=self.config.model.hopfield.heads,
            scaling=self.config.model.hopfield.beta,
            dropout=self.config.model.hopfield.dropout,
        )

        # Initialization
        hopfield_initialization = partial(init_weights, "linear")
        self.hopfield.apply(hopfield_initialization)

    def forward(
        self,
        query_embedding,
        supportActives_embedding,
        supportInactives_embedding,
        supportSetActivesSize,
        supportSetInactivesSize,
        referenceSet_embedding,
    ):
        S = torch.cat(
            [query_embedding, supportActives_embedding, supportInactives_embedding],
            axis=1,
        )

        S_flattend = S.reshape(1, S.shape[0] * S.shape[1], S.shape[2])

        S_h = self.hopfield(
            (referenceSet_embedding, S_flattend, referenceSet_embedding)  # Key  # Query
        )  # Value

        S_updated = S_flattend + S_h
        S_updated_r = S_updated.reshape(S.shape[0], S.shape[1], S.shape[2])

        query_embedding = S_updated_r[:, 0, :]
        query_embedding = torch.unsqueeze(query_embedding, 1)
        supportActives_embedding = S_updated_r[:, 1:13, :]
        supportInactives_embedding = S_updated_r[:, 13:, :]

        return query_embedding, supportActives_embedding, supportInactives_embedding


class FullTransformerPredRetrieval(nn.Module):
    def __init__(self, config: OmegaConf):
        super(FullTransformerPredRetrieval, self).__init__()

        self.config = config

        self.transformer = torch.nn.Transformer(
            d_model=(
                self.config.model.associationSpace_dim
                + self.config.model.transformer.activity_embedding_dim
            ),
            nhead=self.config.model.transformer.number_heads,
            dim_feedforward=self.config.model.transformer.dim_forward,
            dropout=self.config.model.transformer.dropout,
        )

        # Output layer
        self.dropout_o = dropout_mapping[
            config.model.transformer.output_layer.activation
        ](config.model.transformer.output_layer.dropout)
        self.fc_o = nn.Linear(
            self.config.model.associationSpace_dim
            + self.config.model.transformer.activity_embedding_dim,
            1,
        )
        self.act_o = torch.nn.Sigmoid()

    def forward(
        self,
        query_embedding,
        supportActives_embedding,
        supportInactives_embedding,
        supportSetActivesSize,
        supportSetInactivesSize,
    ):
        embedding_dim = supportActives_embedding.shape[2]
        query_embedding = torch.cat(
            [query_embedding, torch.zeros_like(query_embedding[:, :, :64])], dim=2
        )
        supportActives_embedding = torch.cat(
            [
                supportActives_embedding,
                torch.ones_like(supportActives_embedding[:, :, :64]),
            ],
            dim=2,
        )
        supportInactives_embedding = torch.cat(
            [
                supportInactives_embedding,
                (-1.0) * torch.ones_like(supportInactives_embedding[:, :, :64]),
            ],
            dim=2,
        )

        S = torch.cat((supportActives_embedding, supportInactives_embedding), dim=1)
        tgt_key_padding_mask = torch.zeros(S.shape[0], S.shape[1]).to(S.device)

        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                real_actives = supportSetActivesSize[i]
                real_inactives = supportSetInactivesSize[i]

                if j == 0:
                    tgt_key_padding_mask[i, j] = 1
                elif j < 13:
                    if j < (real_actives + 1):
                        tgt_key_padding_mask[i, j] = 1
                else:
                    if j < (real_inactives + 1 + 12):
                        tgt_key_padding_mask[i, j] = 1
        tgt_key_padding_mask = tgt_key_padding_mask.bool()

        S = torch.transpose(S, 0, 1)
        query_embedding = torch.transpose(query_embedding, 0, 1)

        transformer_retrieval = self.transformer(
            S, query_embedding, src_key_padding_mask=tgt_key_padding_mask
        )
        transformer_retrieval = torch.squeeze(
            torch.transpose(transformer_retrieval, 0, 1)
        )

        # FC Prediction Layer
        x = self.dropout_o(transformer_retrieval)
        x = self.fc_o(x)
        x = self.act_o(x)

        return x  # Dimensions should be [Batch-Size, 1]


class HopfieldBlockDTAN(nn.Module):
    def __init__(self, config: OmegaConf):
        super(HopfieldBlockDTAN, self).__init__()

        self.hopfield = Hopfield(
            input_size=config.model.associationSpace_dim,  # R
            hidden_size=config.model.hopfield.dim_QK,  # a_1, Dimension Queries, Keys
            stored_pattern_size=562,  # config.model.encoder.number_hidden_neurons,           # Y
            pattern_projection_size=512,  # config.model.encoder.number_hidden_neurons,       # Y
            output_size=config.model.associationSpace_dim,  # a_2, Dim Values / Dim Dotproduct
            num_heads=config.model.hopfield.heads,
            scaling=config.model.hopfield.beta,
        )

        # Hopfield 2
        # self.hopfield2 = Hopfield(
        #    input_size=512,  # config.model.encoder.number_hidden_neurons,      # R
        #    hidden_size=config.model.hopfield.dim_QK,  # a_1, Dimension Queries, Keys
        #    stored_pattern_size=512,  # config.model.encoder.number_hidden_neurons,           # Y
        #    pattern_projection_size=512,  # config.model.encoder.number_hidden_neurons,       # Y
        #    output_size=config.model.associationSpace_dim,  # a_2, Dim Values / Dim Dotproduct
        #    num_heads=config.model.hopfield.heads,
        #    scaling=config.model.hopfield.beta
        # )

        # Hopfield 3
        # self.hopfield3 = Hopfield(
        #    input_size=512,  # config.model.encoder.number_hidden_neurons,      # R
        #    hidden_size=config.model.hopfield.dim_QK,  # a_1, Dimension Queries, Keys
        #    stored_pattern_size=512,  # config.model.encoder.number_hidden_neurons,           # Y
        #    pattern_projection_size=512,  # config.model.encoder.number_hidden_neurons,       # Y
        #    output_size=config.model.associationSpace_dim,  # a_2, Dim Values / Dim Dotproduct
        #   num_heads=config.model.hopfield.heads,
        #    scaling=config.model.hopfield.beta
        # )

        # Initialization
        hopfield_initialization = partial(init_weights, "linear")
        self.hopfield.apply(hopfield_initialization)

    # def forward(self,R, Y):
    def forward(
        self, QueryEmbeddings, SupportSetActivesEmbedding, SupportSetInactivesEmbedding
    ):
        # Unsqueeze R, Y
        # R = torch.unsqueeze(R, dim=1)

        # Y = torch.unsqueeze(Y, dim=0)

        # Hopfield operation

        # temp
        # SupportSetActivesEmbedding = self.hopfield((SupportSetActivesEmbedding, SupportSetActivesEmbedding,
        #                                            SupportSetActivesEmbedding))
        if SupportSetInactivesEmbedding == "None":
            SupportSetActivesEmbedding = self.hopfield(
                (
                    SupportSetActivesEmbedding,
                    SupportSetActivesEmbedding,
                    SupportSetActivesEmbedding,
                )
            )  # TODO give mask
        else:
            numb_actives = SupportSetActivesEmbedding.shape[
                1
            ]  # ToDo flexibel for ratio
            numb_describtors = SupportSetActivesEmbedding.shape[2]
            SupportSetActivesEmbedding = torch.cat(
                [
                    SupportSetActivesEmbedding,
                    torch.ones_like(SupportSetActivesEmbedding[:, :, :50]),
                ],
                dim=2,
            )
            SupportSetInactivesEmbedding = torch.cat(
                [
                    SupportSetInactivesEmbedding,
                    torch.zeros_like(SupportSetInactivesEmbedding[:, :, :50]),
                ],
                dim=2,
            )

            S_init = torch.cat(
                [SupportSetActivesEmbedding, SupportSetInactivesEmbedding], axis=1
            )
            S_ValuesInput = S_init[:, :, :numb_describtors]
            S = self.hopfield((S_init, QueryEmbeddings, S_ValuesInput))
            # S3 = self.hopfield2((S2, S2, S2))
            # S3 = S2
            # S = self.hopfield3((S3,S3, S_ValuesInput))

            # Todo: second hopfield layer
            # First hopfield: (S,S,S)
            # Second (S,S,S*)

            SupportSetActivesEmbedding = S[:, :numb_actives, :]
            SupportSetInactivesEmbedding = S[:, numb_actives:, :]

        return SupportSetActivesEmbedding, SupportSetInactivesEmbedding


class HopfieldBlockSepp(nn.Module):
    def __init__(self, config: OmegaConf):
        super(HopfieldBlockSepp, self).__init__()

        self.hopfield = Hopfield(
            input_size=562,  # config.model.encoder.number_hidden_neurons,      # R
            hidden_size=config.model.hopfield.dim_QK,  # a_1, Dimension Queries, Keys
            stored_pattern_size=562,  # config.model.encoder.number_hidden_neurons,           # Y
            pattern_projection_size=512,  # config.model.encoder.number_hidden_neurons,       # Y
            output_size=config.model.associationSpace_dim,  # a_2, Dim Values / Dim Dotproduct
            num_heads=config.model.hopfield.heads,
            scaling=config.model.hopfield.beta,
        )

        # Hopfield 2
        # self.hopfield2 = Hopfield(
        #    input_size=512,  # config.model.encoder.number_hidden_neurons,      # R
        #    hidden_size=config.model.hopfield.dim_QK,  # a_1, Dimension Queries, Keys
        #    stored_pattern_size=512,  # config.model.encoder.number_hidden_neurons,           # Y
        #    pattern_projection_size=512,  # config.model.encoder.number_hidden_neurons,       # Y
        #    output_size=config.model.associationSpace_dim,  # a_2, Dim Values / Dim Dotproduct
        #    num_heads=config.model.hopfield.heads,
        #    scaling=config.model.hopfield.beta
        # )

        # Hopfield 3
        # self.hopfield3 = Hopfield(
        #    input_size=512,  # config.model.encoder.number_hidden_neurons,      # R
        #    hidden_size=config.model.hopfield.dim_QK,  # a_1, Dimension Queries, Keys
        #    stored_pattern_size=512,  # config.model.encoder.number_hidden_neurons,           # Y
        #    pattern_projection_size=512,  # config.model.encoder.number_hidden_neurons,       # Y
        #    output_size=config.model.associationSpace_dim,  # a_2, Dim Values / Dim Dotproduct
        #   num_heads=config.model.hopfield.heads,
        #    scaling=config.model.hopfield.beta
        # )

        # Initialization
        hopfield_initialization = partial(init_weights, "linear")
        self.hopfield.apply(hopfield_initialization)

    # def forward(self,R, Y):
    def forward(
        self,
        SupportSetActivesEmbedding,
        SupportSetInactivesEmbedding,
        memory_TrainMols_embedding,
    ):
        # Unsqueeze R, Y
        # R = torch.unsqueeze(R, dim=1)

        # Y = torch.unsqueeze(Y, dim=0)

        # Hopfield operation

        # temp
        # SupportSetActivesEmbedding = self.hopfield((SupportSetActivesEmbedding, SupportSetActivesEmbedding,
        #                                            SupportSetActivesEmbedding))
        if SupportSetInactivesEmbedding == "None":
            SupportSetActivesEmbedding = self.hopfield(
                (
                    SupportSetActivesEmbedding,
                    SupportSetActivesEmbedding,
                    SupportSetActivesEmbedding,
                )
            )  # TODO give mask
        else:
            numb_actives = SupportSetActivesEmbedding.shape[
                1
            ]  # ToDo flexibel for ratio
            numb_describtors = SupportSetActivesEmbedding.shape[2]
            SupportSetActivesEmbedding = torch.cat(
                [
                    SupportSetActivesEmbedding,
                    torch.ones_like(SupportSetActivesEmbedding[:, :, :50]),
                ],
                dim=2,
            )
            SupportSetInactivesEmbedding = torch.cat(
                [
                    SupportSetInactivesEmbedding,
                    -torch.ones_like(SupportSetInactivesEmbedding[:, :, :50]),
                ],
                dim=2,
            )
            memory_TrainMols_embedding = torch.cat(
                [
                    memory_TrainMols_embedding,
                    torch.zeros_like(memory_TrainMols_embedding[:, :, :50]),
                ],
                dim=2,
            )

            S_init = torch.cat(
                [
                    SupportSetActivesEmbedding,
                    SupportSetInactivesEmbedding,
                    memory_TrainMols_embedding,
                ],
                axis=1,
            )
            S_ValuesInput = S_init[:, :, :numb_describtors]
            S = self.hopfield((S_init, S_init, S_ValuesInput))

            # Skip connection
            S = S + S_ValuesInput

            # S3 = self.hopfield2((S2, S2, S2))
            # S3 = S2
            # S = self.hopfield3((S3,S3, S_ValuesInput))

            # Todo: second hopfield layer
            # First hopfield: (S,S,S)
            # Second (S,S,S*)

            SupportSetActivesEmbedding = S[:, :numb_actives, :]
            SupportSetInactivesEmbedding = S[:, numb_actives:, :]

        return SupportSetActivesEmbedding, SupportSetInactivesEmbedding
