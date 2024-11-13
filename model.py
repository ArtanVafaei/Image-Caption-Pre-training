from text.gpt2 import GPT
from vision.resnet import Resnet
import torch
import torch.nn as nn
import sys

class VLM(nn.Module):
    def __init__(self, vision_config, language_config):
        ''' Layers: Resnet, Linear Projection, GPT '''
        super().__init__()
        self.vision_model = Resnet(vision_config) # Resnet
        self.visual_proj = nn.Linear(vision_config.output_embed, language_config.embed_dim) # Linear projection: Image Embd (N, w' * h', image_dim) -> (N, seq_len, text_dim)
        self.language_model = GPT(language_config) # GPT
        self.language_config = language_config

    def forward(self, image, starting_text, labels=None):
        ''' Forward Pass: Resnet (transpose) -> Linear Projection (slice up to seq_len) -> GPT '''
        image = self.vision_model(image).transpose(1, 2)  # Forward pass Resnet, transpose 
        image_embeddings = self.visual_proj(image)[:, :self.language_config.seq_len]  # Linear projection, up to seq_len
        x, loss = self.language_model(image_embeddings, starting_text, labels) # Forward pass GPT, returns logits, loss       
        return x, loss

    # Generates tokens, to decode and view the output in English. Inferencing
    @torch.no_grad()
    def generate(self, image, starting_text, max_new_tokens, temperature=1, do_sample=True, top_k=None):
        ''' Forward Pass: Resnet (transpose) -> Linear Projection (slice up to seq_len) -> GPT generate '''
        image = self.vision_model(image).transpose(1, 2)  # Forward pass Resnet, transpose 
        image_embed = self.visual_proj(image)[:, :self.language_config.seq_len] # Linear projection, up to seq_len
        x, new = self.language_model.generate(image_embed, starting_text, max_new_tokens, temperature, do_sample, top_k) # Generates predicted captions
        return x, new
