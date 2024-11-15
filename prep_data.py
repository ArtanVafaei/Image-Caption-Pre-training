import requests
from PIL import Image
from io import BytesIO
from datasets import load_dataset
import numpy as np
import torch
import tiktoken
import os
import pickle

''' Current Dataset Load: 9569 data points'''

dataset = load_dataset("laion/gpt4v-dataset")['train']

def sample_dataset(upper_bound=1000, seq_len=128, image_size=(224, 224), save_path="dataset/saved"):

    ''' Create empty dataset '''
    # Create empty list of images and captions to load into 
    images = np.zeros((upper_bound,) + image_size + (3,), dtype=np.float32)  # Size (N, W, H, C)
    captions = torch.zeros((upper_bound,) + (seq_len,), dtype=torch.long)  # Size (N, Sequence_length)
    
    # GPT-2 tokenizer for captions
    tokenizer = tiktoken.get_encoding("gpt2")  

    ''' Retrive data from the dataset '''
    i = 0
    for sample in zip(dataset['link'], dataset['caption']):

        # Load up to upper_bound times
        if i >= upper_bound:
            break  

        # Validate correct image
        try:
            # Request data
            response = requests.get(sample[0], timeout=0.5)
            response.raise_for_status()  # Raise an error for bad responses
            image = Image.open(BytesIO(response.content)) # open image from data

            # Preprocess the image to the right format
            image = image.convert("RGB").resize(image_size)
            image = np.array(image)

            # Ensure the image has the correct shape
            if image.shape == (image_size[0], image_size[1], 3):
                images[i] = image  # Store the image in the array
            else:
                print(f"Image from {sample[0]} has an unexpected shape: {image.shape}")
                continue
        except (requests.exceptions.RequestException, Exception) as _: # Fail load, load another data point
            continue

        print(f"Loaded data element {i}")

        # Load image captions 
        tokens = tokenizer.encode_ordinary(sample[1])[:seq_len]
        tokens = tokens + [0] * (seq_len - len(tokens))  # Padding if necessary
        captions[i] = torch.tensor(tokens, dtype=torch.long)

        # Iterate to next data element in our list
        i += 1

    # Save the processed dataset for training
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save the our list
    np.savez_compressed(os.path.join(save_path, "dataset.npz"), images=images, captions=captions)
    print("Dataset saved successfully!")
    return {"images": images, "captions": captions}

# Processes & Saves to dataset [insert dataset size here]
sample_dataset(upper_bound=10000, seq_len=128, image_size=(224, 224)) # Use this, edit upper_bound
