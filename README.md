# Electron-Informed Deep Molecular Representation Learning to Learn Real-World Molecular Physics
Various machine learning methods have been devised to learn molecular representations to accelerate data-driven drug and materials discovery. However, the representation capabilities of existing methods are essentially limited to atom-level information, which is not sufficient to describe real-world molecular physics. Although electron-level information can provide fundamental knowledge about chemical compounds beyond the atom-level information, obtaining the electron-level information about large molecules is computationally impractical. This paper proposes a new method for learning electron-informed molecular representations without additional computation costs by transferring electron-level information about small molecules to large molecules. The proposed method achieved state-of-the-art prediction accuracy on extensive datasets containing experimentally observed molecular physics.

# Run
You can train and evaluate ``GANEIS`` by executing ``train_ganeis.py``.
The evaluation results will be stored in the ``save`` folder.

# Datasets
To reproduce the extrapolation results of SIMD, we should prepare the following two datasets of thermoelectric materials.
- **Starry dataset:** It is a large materials dataset containing thermoelectric materials. Since it was collected by text mining, data pre-processing should be conducted to remove invalid data (reference: https://www.starrydata2.org).
- **ESTM dataset:** It is a refined thermoelectric materials dataset for machine learning. ESTM dataset contains 5,205 experimental observations of thermoelectric materials and their properties (reference: https://doi.org/10.xxxx/xxxxxxxxx).

# Notes
- This repository contains only a subset of the source Starry dataset due to the dataset license. Please visit [Starrydata](https://www.starrydata2.org) to download the full data of the source Starry dataset.
- The full data of the ESTM dataset is provided in the ``dataset folder`` of this repository.
- The ``results folder`` provides the extrapolation results on the full data of the Starry and ESTM dataset. You can check the extrapolation results reported in the paper.
