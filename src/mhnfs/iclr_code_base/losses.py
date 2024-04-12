import torch

def BCE(predictions, labels):
    criterion = torch.nn.BCELoss()

losses = {'BCE': BCE}