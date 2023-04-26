# MHNfs: Context-enriched molecule representations improve few-shot drug discovery

**[Overview](#overview)**
| **[Requirements](#requirements)**
| **[Setup](#setup)**
| **[Model training and evaluation](#model training and evaluation)**
| **[Reproduction of paper results](#reproduction of paper results)**
| **[Data preprocessing](#data preprocessing)**
| **[Citation](#citation)**

**MHNfs** is a few-shot drug discovery model which consists of a **context module**, a **cross-attention module**, and a **similarity module** as described here: https://openreview.net/pdf?id=XrMWUuEevr.


 ![Mhnfs overview](/assets/mhnfs_overview.png)

# Overview
- mhnfs: includes code for the MHNfs modl
- baselines: includes code for the baselines: Frequent Hitters model, Classic Similarity Search, and IterRefLSTM
- preprocessing: includes notebooks for data preprocessing 

## Requirements
If you want to train MHNfs on FS-Mol, you will not a a GPU with approx 20GB RAM.
