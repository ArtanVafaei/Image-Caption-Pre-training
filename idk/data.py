import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
import numpy as np
from datasets import load_dataset

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
    def __init__(self, size, augment=False):
        self.data = self.sample_dataset(size)

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


    def sample_dataset(self, upper_bound):
        # Using numpy arrays and torch tensors for more efficiency 
        images = np.zeros((upper_bound,) + trainer_args.image_size + (3,), dtype=np.float32) # Size (N, W, H, C)
        captions = torch.zeros((upper_bound,) + (trainer_args.seq_len,), dtype=torch.long)  # Size (N, Seqence_length)

        tokenizer = tiktoken.get_encoding("gpt2")       # gpt2 tokenizer for tokenizing captions using gpt2 vocabulary

        i = 0

        for sample in zip(dataset['link'], dataset['caption']):
            if i >= upper_bound:
                break  # Stop if we've filled the array

            try:
                response = requests.get(sample[0], timeout=0.5)
                response.raise_for_status()  # Raise an error for bad responses
                image = Image.open(BytesIO(response.content))
                # preprocessing the image into the right format
                image = image.convert("RGB").resize(trainer_args.image_size)
                image = np.array(image)

                # Ensure the image has the correct shape
                if image.shape == (trainer_args.image_size[0], trainer_args.image_size[1], 3):
                    images[i] = image  # Store the image in the array
                else:
                    print(f"Image from {sample[0]} has an unexpected shape: {image.shape}")
                    continue

            except requests.exceptions.RequestException as e:
                #print(f"Error fetching the image from {sample[0]}: {e}")
                continue
            except Exception as e:
                #print(f"An error occurred while processing the image from {sample[0]}: {e}")
                continue
            
            tokens = tokenizer.encode_ordinary(sample[1])[:trainer_args.seq_len]

            # incase the length of tokens < sequence length, we need to add padding 
            tokens = tokens + [0] * (trainer_args.seq_len - len(tokens))

            captions[i] = torch.tensor(tokens, dtype=torch.long)
            i += 1

        return {"images": images, "captions": captions}
    



        


        


