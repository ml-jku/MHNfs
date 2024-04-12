"""
This file tests whether the AUC and ΔAUC-PR metrics are computed in the expected way.
"""

#---------------------------------------------------------------------------------------
# Dependencies
import pytest
import torch
import pandas as pd
import sys
from src.metrics.performance_metrics import compute_auc_score, compute_dauprc_score

#---------------------------------------------------------------------------------------
# Define tests

class TestMetrics:
    
    def test_compute_auc_score(self):
        
        # Test 1:
        # Single task setting: Perfect classifier
        predictions = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        target_ids = torch.tensor([0, 0, 0, 0, 0, 0])
        
        auc = 1.0
        
        computed_auc = compute_auc_score(predictions, labels, target_ids)[0]
        
        assert auc == computed_auc
        
        # Test 2:
        # Single task setting: Random classifier
        predictions = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        target_ids = torch.tensor([0, 0, 0, 0, 0, 0])
        
        auc = 0.5
        
        computed_auc = compute_auc_score(predictions, labels, target_ids)[0]
        
        assert auc == computed_auc
        
        # Test 4:
        # Multi-task setting: 2 Tasks, assume classifier works perfectly for first task
        # and randomly for the second task
        
        predictions = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 
                                    0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 
                               0, 0, 0, 1, 1, 1])
        target_ids = torch.tensor([0, 0, 0, 0, 0, 0, 
                                  1, 1, 1, 1, 1, 1])
        # shuffle tensors to test on more realistic scenario
        idx = torch.randperm(predictions.shape[0])
        predictions = predictions[idx]
        labels = labels[idx]
        target_ids = target_ids[idx]
        
        aucs = [1.0, 0.5]
        mean_auc = 0.75
        
        computed_mean_auc, computed_aucs, _ = compute_auc_score(predictions, 
                                                                labels, 
                                                                target_ids)
        
        assert mean_auc == computed_mean_auc
        assert aucs == computed_aucs
    
    def test_compute_daucpr_score(self):
        
        # Test 1:
        # Single task setting: Perfect classifier
        predictions = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        target_ids = torch.tensor([0, 0, 0, 0, 0, 0])
        
        dauc_pr = 0.5
        
        computed_dauc_pr = compute_dauprc_score(predictions, labels, target_ids)[0]
        
        assert dauc_pr == computed_dauc_pr
        
        # Test 2:
        # Single task setting: Random classifier
        predictions = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        target_ids = torch.tensor([0, 0, 0, 0, 0, 0])
        
        dauc_pr = 0.0

        computed_dauc_pr = compute_dauprc_score(predictions, labels, target_ids)[0]
        
        assert dauc_pr == computed_dauc_pr
        
        # Test 4:
        # Multi-task setting: 2 Tasks, assume classifier works perfectly for first task
        # and randomly for the second task
        
        predictions = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 
                                    0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 
                               0, 0, 0, 1, 1, 1])
        target_ids = torch.tensor([0, 0, 0, 0, 0, 0, 
                                  1, 1, 1, 1, 1, 1])
        # shuffle tensors to test on more realistic scenario
        idx = torch.randperm(predictions.shape[0])
        predictions = predictions[idx]
        labels = labels[idx]
        target_ids = target_ids[idx]
        
        dauc_prs = [0.5, 0.0]
        mean_dauc_pr = 0.25
        
        computed_mean_dauc_pr, computed_dauc_prs, _ = compute_dauprc_score(predictions, 
                                                                           labels, 
                                                                           target_ids)
        
        assert mean_dauc_pr == computed_mean_dauc_pr
        assert dauc_prs == computed_dauc_prs  
        
        
        

#---------------------------------------------------------------------------------------
# debugging

if __name__ == "__main__":
    test = TestMetrics()
    test.test_compute_auc_score()
    