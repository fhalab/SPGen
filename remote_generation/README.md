This folder contains the majority of the code and data needed to train and generate SPs.

The attention-is-all-you-need-pytorch folder contains a copy of jadore's implementation as used for training and generation.
The repo can be found here: https://github.com/jadore801120/attention-is-all-you-need-pytorch

The code generated for this project can be found in the signal_peptide folder.
An environment yaml file may be found in this directory: transformer_env.yml

The data generated for training is a bit large, and can be found at the Center for Open Science:
https://osf.io/4zjqv/?view_only=47fcb75a57954827a680e3e2e11b3f75
The trained models may also be found at osf: https://osf.io/gmurs/?view_only=3ec60094ee9d4967b22342355d9d2915
and should be downloaded to signal_peptide/outputs/models/model_checkpoints.

Minimal examples are left in the directories.

Within signal_peptide/code/
  1a_analysis.ipynb contains methods for separating sequences by sequence identity.
  1b_splits.ipynb augments and pickles this data.
  2_one_hot_*.ipynb are notebooks for tokenizing the inputs at various * cutoffs for training. 
    This ouput may be found at osf.
  3_pytorch_pairs_training.ipynb contains a sample training loop
    Trained models may be found at osf: https://osf.io/gmurs/?view_only=3ec60094ee9d4967b22342355d9d2915
  4_testgeneration.ipynb contains an example notebook for generating SPs given an input excel file.
    



