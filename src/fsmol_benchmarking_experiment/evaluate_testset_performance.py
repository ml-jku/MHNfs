"""
Performance evaluation of the MHNfs model on the FS-Mol test set 
"""

# --------------------------------------------------------------------------------------
# Global variables
DEVICE = "cuda"
NUM_THREADS = 12

# --------------------------------------------------------------------------------------
# Libraries
import os
import torch
import pandas as pd
from hydra import compose, initialize
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
import sys
sys.path.append('../../../MHNfs/')
from src.mhnfs.models import MHNfs
from src.metrics.performance_metrics import compute_auc_score, compute_dauprc_score
from src.data.dataloader import FSMolDataModule


# --------------------------------------------------------------------------------------
# Functions

def evaluate_performance(model, test_dataloader, device):
    all_preds = list()
    all_labels = list()
    all_target_ids = list()
    
    print("... make predictions ...")
    for batch in test_dataloader:
        # Prepare inputs
        query_embedding = batch["queryMolecule"]
        labels = batch["label"].squeeze().float()
        support_actives_embedding_padded = batch["supportSetActives"]
        support_inactives_embedding_padded = batch["supportSetInactives"]
        target_ids = batch["taskIdx"]

        support_actives_size = batch["supportSetActivesSize"]
        support_inactives_size = batch["supportSetActivesSize"]

        support_actives_mask = torch.cat(
            [
                torch.cat(
                    [torch.tensor([False] * d), torch.tensor([True] * (12 - d))]
                ).reshape(1, -1)
                for d in support_actives_size
            ],
            dim=0,
        )
        support_inactives_mask = torch.cat(
            [
                torch.cat(
                    [torch.tensor([False] * d), torch.tensor([True] * (12 - d))]
                ).reshape(1, -1)
                for d in support_inactives_size
            ],
            dim=0,
        )

        # Make predictions
        preds = (
            model(
                query_embedding.to(device),
                support_actives_embedding_padded.to(device),
                support_inactives_embedding_padded.to(device),
                support_actives_size.to(device),
                support_inactives_size.to(device),
                support_actives_mask.to(device),
                support_inactives_mask.to(device),
            )
            .detach()
            .cpu()
            .float()
        )

        # Store batch outcome
        all_preds.append(preds)
        all_labels.append(labels)
        all_target_ids.append(target_ids)

    predictions = torch.cat([p for p in all_preds], axis=0)
    labels = torch.cat([l for l in all_labels], axis=0)
    target_ids = torch.cat([t for t in all_target_ids], axis=0)

    # Compute metrics
    print("... compute AUC and  ΔAUC-PR ...")
    auc, aucs, target_ids_store = compute_auc_score(predictions, labels, target_ids)
    d_auc_pr, daucprs, _ = compute_dauprc_score(predictions, labels, target_ids)
    
    # Create results dataframe
    df = pd.DataFrame({'auc':aucs, 'daucprs':daucprs, 'target':target_ids_store})

    
    return auc, d_auc_pr, df

def run_evaluation_script(device=DEVICE, num_threads=NUM_THREADS):
    
    # Set number of threads
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    
    # Load model
    print("Load model ...")
    model = MHNfs.load_from_checkpoint(
        sys.path[-1] + 'assets/mhnfs_data/mhnfs_checkpoint.ckpt')
    model.eval()
    seed_everything(1234)
    model = model.to(device)
    model._update_context_set_embedding()

    # Load datamodule
    print("Load datamodule ...")
    seed_everything(1234)
    with initialize(config_path="../mhnfs/configs/", job_name=None):
        cfg = compose(config_name="cfg")
    dm = FSMolDataModule(cfg)
    dm.setup()
    dataloader = dm.test_dataloader()

    # Evaluation on testset
    print("Evaluate on test set ...")
    auc, d_auc_pr, _ = evaluate_performance(model, dataloader, device)
    print(f"Mean     AUC: {auc}")
    print(f"Mean ΔAUC-PR: {d_auc_pr}")


# ---------------------------------------------------------------------------------------
# Execute script
if __name__ == "__main__":
    run_evaluation_script()
