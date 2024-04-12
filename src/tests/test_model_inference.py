"""
This file tests whether the model predictions for MHNfs match the predictions made on
the JKU development server (varified model, server conda env with spec. packages ...)

Here the MHNfs code base for inference (single task, no padding, ...) is tested.
"""

#---------------------------------------------------------------------------------------
# Dependencies
import pytest
import torch
import pandas as pd
from src.mhnfs.inference.load_trained_model import (load_trained_model,
                                                    MHNfs_inference_module)

#---------------------------------------------------------------------------------------
# Define tests

class TestTrainedMHNfs:
    
    def test_mhnfs_prediction_on_cpu(self, model_input_query,
                                     model_input_support_actives,
                                     model_input_support_inactives,
                                     model_predictions):
        
        # Load model
        mhnfs = load_trained_model()
        
        # Provide support set sizes
        support_actives_size = torch.tensor(model_input_support_actives.shape[1])
        support_inactives_size = torch.tensor(model_input_support_inactives.shape[1])
        
        # Make predictions
        predictions = mhnfs(
            model_input_query,
            model_input_support_actives,
            model_input_support_inactives,
            support_actives_size,
            support_inactives_size
        ).detach()
        
        # Compare predictions
        assert torch.allclose(predictions, model_predictions, atol=0.01, rtol=0.)
    
    def test_mhnfs_prediction_on_gpu(self, model_input_query,
                                     model_input_support_actives,
                                     model_input_support_inactives,
                                     model_predictions):
        
        # Load model
        mhnfs = load_trained_model().to('cuda')
        mhnfs.context_embedding = mhnfs.context_embedding.to('cuda')
        
        # Provide support set sizes
        support_actives_size = torch.tensor(model_input_support_actives.shape[1])
        support_inactives_size = torch.tensor(model_input_support_inactives.shape[1])
        
        # Make predictions
        predictions = mhnfs(
            model_input_query.to('cuda'),
            model_input_support_actives.to('cuda'),
            model_input_support_inactives.to('cuda'),
            support_actives_size.to('cuda'),
            support_inactives_size.to('cuda')
        ).detach().cpu()
        
        # Compare predictions
        assert torch.allclose(predictions, model_predictions, atol=0.01, rtol=0.)   

class Test_MHNfs_inference_module:
    
    def test_predictions_on_cpu(self,
                                model_input_query,
                                model_input_support_actives,
                                model_input_support_inactives,
                                model_predictions):
        # Load model
        mhnfs_inference_module = MHNfs_inference_module(device='cpu')
        
        # Make predictions
        predictions = mhnfs_inference_module.predict(
            model_input_query,
            model_input_support_actives,
            model_input_support_inactives
        )
        
        # Compare predictions
        assert torch.allclose(predictions, model_predictions, atol=0.01, rtol=0.)
    
    def test_predictions_on_gpu(self,
                                model_input_query,
                                model_input_support_actives,
                                model_input_support_inactives,
                                model_predictions):
        # Load model
        mhnfs_inference_module = MHNfs_inference_module(device='gpu')
        
        # Make predictions
        predictions = mhnfs_inference_module.predict(
            model_input_query,
            model_input_support_actives,
            model_input_support_inactives
        )
        
        # Compare predictions
        assert torch.allclose(predictions, model_predictions, atol=0.01, rtol=0.)
    