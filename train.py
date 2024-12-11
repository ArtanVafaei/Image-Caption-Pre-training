import torch
import torch.nn as nn
import numpy as np

from config import parameters
from data_load import getDataSet
from model_load import getResNet, getGPT2LMHeadModel
from model import ImageCaptioningModel
from transformers import AdamW

import random
from model_load import tokenizer
from test import generate_caption
from evaluate import load

def train(resnet_model, gpt2lmhead_model, pretrained, dataset):

    ''' Load in parameters/hyperparameters'''
    args = parameters
    batch_size = args.batch_size
    num_epochs = args.max_iters
    print_every = args.eval_interval
    device = args.device
    learning_rate = args.learning_rate
    bertscore = load("bertscore")

    ''' Create and split dataset '''
    train_dataloader, val_dataloader = getDataSet(batch_size, dataset)

    ''' Load in ResNet and GPT2 model '''
    resnet = getResNet(version=resnet_model, pretrained=pretrained).to(device)
    resnet = nn.Sequential(*list(resnet.children())[:-2]) # Remove avg pool and fc layer
    gpt2 = getGPT2LMHeadModel(version=gpt2lmhead_model, pretrained=pretrained).to(device)

    ''' Create Model with optimizer '''
    model = ImageCaptioningModel(resnet, resnet_model, gpt2).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    ''' Create loss list and BertScore lists '''
    training_losses = np.zeros((num_epochs), dtype=np.float32)
    validation_losses = np.zeros((num_epochs), dtype=np.float32)
    precision_bc = np.zeros((num_epochs), dtype=np.float32)
    recall_bc = np.zeros((num_epochs), dtype=np.float32)
    f1_bc = np.zeros((num_epochs), dtype=np.float32)

    ''' Train the model '''
    for epoch in range(num_epochs):

        # Train model
        model.train()
        total_loss = 0
        iteration_loss = 0

        for idx, batch in enumerate(train_dataloader, 1):
            batch = tuple(t.to(device) for t in batch)
            images, input_ids, masks = batch

            optimizer.zero_grad()

            outputs = model(images, input_ids, attention_mask = masks)
            loss = outputs.loss

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            iteration_loss += loss.item()

            if idx % print_every == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} | Iteration {idx}/{len(train_dataloader)} | "
                    f"Training Loss: {iteration_loss / print_every:.3f}")
                iteration_loss = 0

        # Esimate_loss    
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = tuple(t.to(device) for t in batch)
                images, input_ids, masks = batch
                outputs = model(images, input_ids, attention_mask=masks)
                loss = outputs.loss
                val_loss += loss.item()
        val_loss = val_loss / len(val_dataloader)
        
        # Save both the training loss mean and validation loss mean
        training_losses[epoch] = total_loss / len(train_dataloader)
        validation_losses[epoch] = val_loss

        ''' Calculate BertScore - Precision, Recall, F1 Score'''

        # Retrieve Random Image
        random_idx = random.randint(0, len(val_dataloader) - 1)
        images, captions, _ = next(iter(val_dataloader))
        image = images[random_idx].to(device)

        # Retrieve associating caption
        encoded_caption = captions[random_idx]
        reference_caption = tokenizer.decode(encoded_caption, skip_special_token=True) # decode

        # Generate Caption
        prediected_caption = generate_caption(image, model, tokenizer)

        # Generate/save the bertscore
        result = bertscore.compute(predictions=[prediected_caption], references=[reference_caption], model_type="distilbert-base-uncased")
        precision = result.get('precision')[0]
        recall = result.get('recall')[0]
        f1 = result.get('f1')[0]
        precision_bc[epoch] = precision
        recall_bc[epoch] = recall
        f1_bc[epoch] = f1

        print(f"Epoch {epoch + 1}/{num_epochs} | Training Loss: {total_loss / len(train_dataloader):.3f} | "
        f"Validation Loss: {val_loss:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1 Score: {f1:.3f}\n")


    ''' Return model and data for generating results'''
    return model, val_dataloader, training_losses, validation_losses, precision_bc, recall_bc, f1_bc