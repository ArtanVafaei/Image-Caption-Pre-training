import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import random
import kagglehub
import torch
from torch.utils.data import DataLoader

from model_load import tokenizer


''' Collate function for dataloader '''
def collate_fn(data):
    images, captions = zip(*data)
    images = torch.stack(images, 0)

    captions = [[tokenizer.bos_token_id] + cap + [tokenizer.eos_token_id] for cap in captions]

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths), dtype=torch.long)
    # Create attention masks
    masks = torch.zeros(len(captions), max(lengths), dtype=torch.long)

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = torch.LongTensor(cap)
        masks[i, :end] = 1

    return images, targets, masks

''' Dataset loader for the training model '''
# Load in dataset and get file path
flickr8k_path = kagglehub.dataset_download('adityajn105/flickr8k')

''' Use the Flickr8kDataset '''
# Define the transformation for the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit most pre-trained models
    transforms.ToTensor(),
])

class Flickr8kDataset(Dataset):
    def __init__(self, annotations_file, img_dir, tokenizer, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_captions = pd.read_csv(annotations_file)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.img_captions)//5

    def __getitem__(self, idx):
        file_name = self.img_captions.iloc[5*idx, 0]
        img_path = os.path.join(self.img_dir, file_name)
        image = Image.open(img_path)
        caption = random.choice(self.img_captions.iloc[5*idx : 5*(idx+1), 1].tolist())
        tokenized_caption = self.tokenizer.encode(caption)
        if self.transform:
            image = self.transform(image)
        return image, tokenized_caption



def getDataSet(batch_size, dataset):

    # Retrive image folder directory and caption file directory
    if dataset == "dataset1":
        img_dir = flickr8k_path + r"\images"
        annotations_file = flickr8k_path + r"\captions.txt"

    # Create and split dataset 
    dataset = Flickr8kDataset(annotations_file=annotations_file, img_dir=img_dir, tokenizer=tokenizer, transform=transform)

    split_point = int(0.9*len(dataset))

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [split_point, len(dataset) - split_point])

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    return train_dataloader, val_dataloader

def getDataSetName(dataset):
    if dataset == "dataset1":
        return "Flickr8K"