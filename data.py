import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
import numpy as np
from datasets import load_dataset
import os
import requests
from PIL import Image
from io import BytesIO
import tiktoken
from config import trainer_config

trainer_args = trainer_config.trainer_args

'''
uses an image-text pair dataset from huggingface: https://huggingface.co/datasets/laion/gpt4v-dataset
'''
dataset = load_dataset("laion/gpt4v-dataset")['train']

# object that will contain the dataset as well as functions to help handle it
class ImageTextDataset(Dataset):
    def __init__(self, size, train, augment=False):
        if train:
            self.data = self.sample_dataset(size)
        else:
            self.data = self.test_sample_dataset(size)

    # when len(self) is called, it returns the length of the number of samples the dataset contains
    def __len__(self):
        return len(self.data["images"])
    
    # returns the images and corresponding captions in idx position where idx is positions from i...i+batch_size-1
    def __getitem__(self, idx):
        return self.transform(self.data["images"][idx]), self.data["captions"][idx]
    
    def transform(self, images):
        # Normalize the entire batch and convert to a PyTorch tensor
        images = images / 255.0  # Normalize to [0, 1]
        images = torch.tensor(images, dtype=torch.float32)  # Convert to PyTorch tensor
        images = images.permute(2, 0, 1)  # Change to (batch_size, channels, height, width)
        return images


    def sample_dataset(self, upper_bound, save_path="dataset/saved"):
        data = np.load(os.path.join(save_path, "dataset.npz"))
        images = data['images']
        captions = torch.tensor(data['captions'], dtype=torch.long)
        print("Dataset loaded successfully!")
        return {"images": images[:upper_bound], "captions": captions[:upper_bound]}
    
    def test_sample_dataset(self, upper_bound, save_path="dataset/saved"):
        data = np.load(os.path.join(save_path, "dataset.npz"))
        images = data['images']
        captions = torch.tensor(data['captions'], dtype=torch.long)
        print("Dataset loaded successfully!")
        return {"images": images[upper_bound: upper_bound+trainer_args.test_dataset_size], "captions": captions[upper_bound: upper_bound+trainer_args.test_dataset_size]}

    



        


        


