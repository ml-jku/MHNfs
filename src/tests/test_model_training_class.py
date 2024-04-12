"""
This file tests whether the model predictions for MHNfs match the predictions made on
the JKU development server (varified model, server conda env with spec. packages ...)

Here, the MHNfs code base for training (multi-task, padding, ...) is tested.
"""

#---------------------------------------------------------------------------------------
# Dependencies
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

#---------------------------------------------------------------------------------------
# Define tests

class TestMHNfs:
    
    def test_mhnfs(self,
                   model_input_query,
                   model_input_support_actives,
                   model_input_support_inactives,
                   model_predictions,
                   model_trainingClass):
        
        # Load model
        pl.seed_everything(1234)
        model = model_trainingClass.to('cuda')
        model.eval()
        model._update_context_set_embedding()
        
        model = model.to('cuda')
        model.context_embedding = model.context_embedding.to('cuda')
        
        # Provide support set sizes
        support_actives_size = torch.tensor(model_input_support_actives.shape[1])
        support_inactives_size = torch.tensor(model_input_support_inactives.shape[1])
        
        # Support set padding
        model_input_support_actives = torch.cat(
            [model_input_support_actives,
             torch.zeros(model_input_support_actives.shape[0],
                         12 - model_input_support_actives.shape[1],
                         model_input_support_actives.shape[2])],
            dim=1
        )
        model_input_support_inactives = torch.cat(
            [model_input_support_inactives,
             torch.zeros(model_input_support_inactives.shape[0],
                         12 - model_input_support_inactives.shape[1],
                         model_input_support_inactives.shape[2])],
            dim=1
        )
        
        # Expand support set
        model_input_support_actives = model_input_support_actives.expand(
                                                    model_input_query.shape[0], -1, -1)
        model_input_support_inactives = model_input_support_inactives.expand(
                                                    model_input_query.shape[0], -1, -1)
        support_actives_size = support_actives_size.expand(
                                                    model_input_query.shape[0])
        support_inactives_size = support_inactives_size.expand(
                                                    model_input_query.shape[0])
        
        support_molecules_active_mask = torch.cat(
            [
                torch.cat(
                    [torch.tensor([False] * d), torch.tensor([True] * (12 - d))]
                ).reshape(1, -1)
                for d in support_actives_size
            ],
            dim=0,
        ).to('cuda')
        support_molecules_inactive_mask = torch.cat(
            [
                torch.cat(
                    [torch.tensor([False] * d), torch.tensor([True] * (12 - d))]
                ).reshape(1, -1)
                for d in support_inactives_size
            ],
            dim=0,
        ).to('cuda')
        
        # Make predictions
        predictions = model(
            model_input_query.to('cuda'),
            model_input_support_actives.to('cuda'),
            model_input_support_inactives.to('cuda'),
            support_actives_size.to('cuda'),
            support_inactives_size.to('cuda'),
            support_molecules_active_mask.to('cuda'),
            support_molecules_inactive_mask.to('cuda')
        ).detach().cpu()
        
        # Compare predictions
        assert torch.allclose(predictions, model_predictions, atol=0.01, rtol=0.)
        

#---------------------------------------------------------------------------------------
# debugging
if __name__ == "__main__":
    # Test
    test = TestMHNfs()
    test.test_mhnfs()