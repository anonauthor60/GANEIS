# Electron-Informed Deep Molecular Representation Learning to Learn Real-World Molecular Physics
Various machine learning methods have been devised to learn molecular representations to accelerate data-driven drug and materials discovery. However, the representation capabilities of existing methods are essentially limited to atom-level information, which is not sufficient to describe real-world molecular physics. Although electron-level information can provide fundamental knowledge about chemical compounds beyond the atom-level information, obtaining the electron-level information about large molecules is computationally impractical. This paper proposes a new method for learning electron-informed molecular representations without additional computation costs by transferring electron-level information about small molecules to large molecules. The proposed method achieved state-of-the-art prediction accuracy on extensive datasets containing experimentally observed molecular physics.

# Run
You can train and evaluate ``GANEIS`` by executing ``train_ganeis.py``.

# Notes
- The evaluation results will be stored in the ``save`` folder.
- Due to the file size, the pre-converted datasets have been compressed to ``.zip`` file. Please, decompress the ``.zip`` files in ``save/datasets`` folder.
- You can use pre-converted datasets by setting ``decomposition`` to ``False`` in ``train_ganeis.py``.
