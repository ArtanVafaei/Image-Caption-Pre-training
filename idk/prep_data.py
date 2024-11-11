import requests
from PIL import Image
from io import BytesIO
from datasets import load_dataset
import numpy as np
import torch
import tiktoken
import os
import pickle

dataset = load_dataset("laion/gpt4v-dataset")['train']

def sample_dataset(upper_bound=1000, seq_len=128, image_size=(224, 224), save_path="dataset/saved"):
    images = np.zeros((upper_bound,) + image_size + (3,), dtype=np.float32)  # Size (N, W, H, C)
    captions = torch.zeros((upper_bound,) + (seq_len,), dtype=torch.long)  # Size (N, Sequence_length)
    
    tokenizer = tiktoken.get_encoding("gpt2")  # GPT-2 tokenizer for captions

    i = 0
    for sample in zip(dataset['link'], dataset['caption']):
        if i >= upper_bound:
            break  # Stop if we've filled the array

        try:
            response = requests.get(sample[0], timeout=0.5)
            response.raise_for_status()  # Raise an error for bad responses
            image = Image.open(BytesIO(response.content))
            # Preprocess the image to the right format
            image = image.convert("RGB").resize(image_size)
            image = np.array(image)

            # Ensure the image has the correct shape
            if image.shape == (image_size[0], image_size[1], 3):
                images[i] = image  # Store the image in the array
            else:
                print(f"Image from {sample[0]} has an unexpected shape: {image.shape}")
                continue

        except requests.exceptions.RequestException as e:
            continue
        except Exception as e:
            continue
        
        tokens = tokenizer.encode_ordinary(sample[1])[:seq_len]
        tokens = tokens + [0] * (seq_len - len(tokens))  # Padding if necessary

        captions[i] = torch.tensor(tokens, dtype=torch.long)
        i += 1

    # Save the processed dataset for future use
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.savez_compressed(os.path.join(save_path, "dataset.npz"), images=images, captions=captions)
    print("Dataset saved successfully!")
    return {"images": images, "captions": captions}

# Example usage to process and save the dataset
sample_dataset(upper_bound=10000, seq_len=128, image_size=(224, 224))
