# MHNfs: Context-enriched molecule representations improve few-shot drug discovery

**MHNfs** is a few-shot drug discovery model which consists of a **context module**, a **cross-attention module**, and a **similarity module** as described here: https://openreview.net/pdf?id=XrMWUuEevr.


 ![Mhnfs overview](/assets_github/mhnfs_overview.png)

## 👨‍👧‍👦 MHNfs repository family
- **This repo**: <br>
    This repo is helpful, if you want to train and evaluate MHNfs on FS-Mol. Adaption to other datasets should be quite simple as long as the new data is preprocessed in the way FS-Mol was preprocessed. This repo includes:  
    * Code to train and evaluate MHNfs on the benchmark experiment FS-Mol.
    * Code for data preprocessing
    * Competitors (Neural Similarity Search, ProtoNet, IterRefLSTM) #TODO
- **Huggingface / Streamlit repo**:<br>
    We recommend this repo, if you want to run the trained MHNfs model in inference mode.
    * MHNfs was trained on FS-Mol
    * The inference pipeline can be called from terminal or via the web interface.
    * No separate data preprocessing is needed since this step was already included into the inference pipeline. You'll just need to provided molecules in SMILES notation.
    * Link: <url>https://huggingface.co/spaces/ml-jku/mhnfs/blob/main/README.md</url>
- **Autoregressive inference repo**:<br>
    We recommend this repo, if you would like to explore potentials to improve MHNfs predictions iteratively while inference.
    * Preliminary and ongoing research topic: Effectiveness highly dependent from setting and predictions can be impaired for unsuitable tasks
    * Link: <url>https://github.com/ml-jku/autoregressive_activity_prediction</url>

## 🚀 Get started with this repo

### 📀 Clone repo and install requirements
```bash
# Clone repo
git clone https://github.com/ml-jku/MHNfs.git

# Navigate into repo folder
cd ./MHNfs

# Download and unzip assets folder (~600 MB zipped, ~4 GB unzipped)
pip install gdown
gdown https://drive.google.com/uc?id=1M6NQWDOBGPSxxvmG7y7kOeumnq_ccSvg
unzip assets.zip

# Create conda environment
conda env create -f env.yml -n your_env_name
conda activate your_env_name
```

### ✅ Evaluate trained MHNfs on FS-Mol
```bash
# Move into experiment folder
cd ./src/fsmol_benchmarking_experiment/

# Run evaluation script
python evaluate_testset_performance.py
```

### 〽️ Train MHNfs
**Load preprocessed FS-Mol data**
```bash
# Move to location at which data should be stored
cd path_to_preprocessed_fsmol_data_dir

# Download and unzip data (~400 MB zipped, ~5 GB unzipped)
gdown https://drive.google.com/uc?id=1SEi8dkkdXudWzRFAYABBckk12tNWfGtX
unzip preprocessed_data
```

**Update data path in config**
```yaml
# config location: .src/mhnfs/configs/system/system.yaml

data:
  path: "path_to_preprocessed_fsmol_data_dir" #TODO change
  dir_training: "training/"
  dir_validation: "validation/"
  dir_test: "test/"
...
```

**Train MHNfs**
```bash
# Move into experiment folder
cd ./src/fsmol_benchmarking_experiment/

# Run training script
python train.py
```

## 📑 Rerun data preprocessing

**Clone FS-Mol repo**
```bash
# Move to location at which the fs-mol repo should be stored
cd path_to_fsmol_repo_dir

# Clone repo
git clone https://github.com/microsoft/FS-Mol.git
```
**Perform preprocessing**:
* Use preprocessing notebook: ```.src/data/preprocessing/preprocessing.ipynb```
* Adapt paths to the raw FS-Mol data (flagged with ```#TODO```)

## ✨ ICLR models
The code base has developed in the time after the ICLR conference (also the MHNfs model has changed slightly). To use the models presented in the manuscript, the checkpoints need to be downloaded and a older code base has to be used.

**Download old checkpoints**
```bash
# Move into assets and create "old checkpoints" folder
cd ./assets/
mkdir old_checkpoints
cd old_checkpoints

# Downlad checkoints
# mhnfs
gdown https://drive.google.com/uc?id=1IdhnGC-Kq5MuiAvV737jJepWQYqZVqIJ
unzip mhnfs

#iterRefLstm
gdown https://drive.google.com/uc?id=12LjxW87lA7YjH4-nYXle7EN_ZfFSMvcS
unzip iterref
```
**Load models from checkpoints in your python script**
```python
import sys
sys.path.append(".")
from src.mhnfs.iclr_code_base.models import MHNfs, IterRef

# Load one of the mhnfs checkpoints
mhnfs_iclr = MHNfs.load_from_checkpoint(
    ".mhnfs/epoch=94-step=19855.ckpt")

# Load one of the iterref checkpoints
iterref = IterRef.load_from_checkpoint(
    ".iterref/epoch=35-step=7524.ckpt")
```

## 📚 Cite us
@inproceedings{
    schimunek2023contextenriched,
    title={Context-enriched molecule representations improve few-shot drug discovery},
    author={Johannes Schimunek and Philipp Seidl and Lukas Friedrich and Daniel Kuhn and Friedrich Rippmann and Sepp Hochreiter and Günter Klambauer},
    booktitle={The Eleventh International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=XrMWUuEevr}
}
