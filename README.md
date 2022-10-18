# MHNfs: Context-enriched molecule representations improve few-shot drug discovery

**MHNfs** is a few-shot drug discovery model which consists of a **context module**, a **cross-attention module**, and a **similarity module**.

 ![Mhnfs overview](/assets/mhnfs_overview.png)
 
 A central task in computational drug discovery is to construct models from known active molecules to find further promising molecules for subsequent screening. However, typically only very few active molecules are known. Therefore, few-shot learning methods have the potential to improve the effectiveness of this critical phase of the drug discovery process. We introduce a new method for few-shot drug discovery. Its main idea is to enrich a molecule representation by knowledge about known context or reference molecules. Our novel concept for molecule representation enrichment is to associate molecules from both the support set and the query set with a large set of reference (context) molecules through a modern Hopfield network. Intuitively, this enrichment step is analogous to a human expert who would associate a given molecule with familiar molecules whose properties are known. The enrichment step reinforces and amplifies the covariance structure of the data and simultaneously removes spurious correlations arising from the decoration of molecules. We analyze our novel method on FS-Mol, which is the only established few-shot learning benchmark dataset for drug discovery. An ablation study shows that the enrichment step of our method is key to improving the predictive quality. In a domain shift experiment, our new method is more robust than other methods. On FS-Mol, our new method achieves a new state-of-the-art and outperforms all other few-shot methods.
