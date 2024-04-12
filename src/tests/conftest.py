"""
Needed objects for tests
"""

#---------------------------------------------------------------------------------------
# Dependencies
import pytest
import pandas as pd
import pickle
import numpy as np
import torch
from src.mhnfs.models import MHNfs

#---------------------------------------------------------------------------------------
# Define fixtures

#---------------------------------------------------------------------------------------
# Model
@pytest.fixture(scope="session")
def model_input_query():
    current_loc = __file__.rsplit("/",3)[0]
    model_input_query = torch.load(
        current_loc + "/assets/test_reference_data/model_input_query.pt")
    return model_input_query

@pytest.fixture(scope="session")
def model_input_support_actives():
    current_loc = __file__.rsplit("/",3)[0]
    model_input_support_actives = torch.load(
        current_loc + "/assets/test_reference_data/model_input_support_actives.pt")
    return model_input_support_actives

@pytest.fixture(scope="session")
def model_input_support_inactives():
    current_loc = __file__.rsplit("/",3)[0]
    model_input_support_inactives = torch.load(
        current_loc + "/assets/test_reference_data/model_input_support_inactives.pt")
    return model_input_support_inactives

@pytest.fixture(scope="session")
def model_predictions():
    current_loc = __file__.rsplit("/",3)[0]
    model_predictions = torch.load(
        current_loc + "/assets/test_reference_data/model_predictions.pt")
    return model_predictions

@pytest.fixture(scope="session")
def model_trainingClass():
    current_loc = __file__.rsplit("/",3)[0]
    model = MHNfs.load_from_checkpoint(
        current_loc + '/assets/mhnfs_data/mhnfs_checkpoint.ckpt')
    return model

#---------------------------------------------------------------------------------------