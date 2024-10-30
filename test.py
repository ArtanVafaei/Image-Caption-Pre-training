# import numpy as np 

# # Load the arrays from the .npz file
# data = np.load("dataset/saved")

# # Access the arrays using the keys
# train_array = data['train']
# test_array = data['test']

# print(train_array[0])
# import torch
# import os

# checkpoint = {
#     'layer': torch.nn.Linear(1, 1)  # Corrected 'torch.nn.linear' to 'torch.nn.Linear'
# }

# # Ensure the current directory exists (this is usually not needed since '.' refers to the current directory)
# output_dir = "out"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir, exist_ok=True)

# # Save the checkpoint
# torch.save(checkpoint, os.path.join(output_dir, 'ckpt.pt'))
