# End-to-End Pre-training for Image Captioning

- If you have cloned the repository, please check the "How to use" section
- For Proposal Detail, check Proposal.pdf

# How to use our program

# Hyperparameters Configuration
There are 3 different files present in the config folder to edit your hyperparameters:
    gpt2_config.py      
    resent_config.py
    trainer_config.py

# Dataset Pre-loading
The program implements pre-loading from a GPT4 dataset for our program to train over.

To properly pre-load, run:
    python prep_data.py

After pre-loading, a new file in the dataset\saved folder should show up called dataset.npz

# Model Process
Our model integrates a Resnet, Linear Projection, and a GPT model to predict image captions





