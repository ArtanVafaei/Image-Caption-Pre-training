import torch
import numpy as np
import os

def load_saved_dataset(save_path="dataset/saved"):
    data = np.load(os.path.join(save_path, "dataset.npz"))
    images = data['images']
    captions = torch.tensor(data['captions'], dtype=torch.long)
    print("Dataset loaded successfully!")
    return {"images": images, "captions": captions}

# Example usage to load the saved dataset
loaded_data = load_saved_dataset()

for b in loaded_data:
    print(loaded_data[b][1])