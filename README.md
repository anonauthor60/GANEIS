# Electron-Informed Deep Molecular Representation Learning to Learn Real-World Molecular Physics

# Run
You can train and evaluate ``GANEIS`` by executing ``train_ganeis.py``.
The evaluation results will be stroed in the ``save`` folder.

# Datasets
To reproduce the extrapolation results of SIMD, we should prepare the following two datasets of thermoelectric materials.
- **Starry dataset:** It is a large materials dataset containing thermoelectric materials. Since it was collected by text mining, data pre-processing should be conducted to remove invalid data (reference: https://www.starrydata2.org).
- **ESTM dataset:** It is a refined thermoelectric materials dataset for machine learning. ESTM dataset contains 5,205 experimental observations of thermoelectric materials and their properties (reference: https://doi.org/10.xxxx/xxxxxxxxx).

# Notes
- This repository contains only a subset of the source Starry dataset due to the dataset license. Please visit [Starrydata](https://www.starrydata2.org) to download the full data of the source Starry dataset.
- The full data of the ESTM dataset is provided in the ``dataset folder`` of this repository.
- The ``results folder`` provides the extrapolation results on the full data of the Starry and ESTM dataset. You can check the extrapolation results reported in the paper.
