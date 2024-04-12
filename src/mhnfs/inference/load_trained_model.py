#---------------------------------------------------------------------------------------
# Dependencies
import pytorch_lightning as pl
import torch
from src.mhnfs.inference.trained_model_singletask import MHNfs

#---------------------------------------------------------------------------------------
# Define function

def load_trained_model():
    """
    This function loads a trained MHNfs model from a checkpoint file.
        * Training on FS-Mol training set
        * Hyperparameter search on FS-Mol validation set 
    """
    
    pl.seed_everything(1234)
    current_loc = __file__.rsplit("/",4)[0]
    model = MHNfs.load_from_checkpoint(current_loc +
                                       "/assets/mhnfs_data/mhnfs_checkpoint.ckpt")
    model._update_context_set_embedding()
    model.eval()
    
    return model 

class MHNfs_inference_module:
    """
    This module is a wrapper for the pre-trained MHNfs model and is suppesed to be used
    for inference.
    """
    def __init__(self, device:['cpu', 'gpu']='cpu'):
        
        if device == 'cpu':
            self.device = device
        elif device == 'gpu':
            self.device = 'cuda'
        
        # Load model
        self.model = load_trained_model()
        
        # Move to GPU if requested
        if device == 'gpu':
            self.model = self.model.to('cuda')
            self.model.context_embedding = self.model.context_embedding.to('cuda')
    
    def predict(self, query_tensor, support_actives_tensor, support_inactives_tensor):
        """
        This function creates support-set-size tensor and feeeds all tensors into the
        model forward function.
        """
        
        # Create support set size tensors
        support_actives_size = torch.tensor(support_actives_tensor.shape[1])
        support_inactives_size = torch.tensor(support_inactives_tensor.shape[1])
        
        # Make predictions
        predictions = self.model(
            query_tensor.to(self.device),
            support_actives_tensor.to(self.device),
            support_inactives_tensor.to(self.device),
            support_actives_size.to(self.device),
            support_inactives_size.to(self.device)
        ).detach().cpu()
        
        return predictions        
        
        
        
        