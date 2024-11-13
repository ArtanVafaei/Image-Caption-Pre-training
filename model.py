from text.gpt2 import GPT
from vision.resnet import Resnet
import torch
import torch.nn as nn

class VLM(nn.Module):
    def __init__(self, vision_config, language_config):
        super().__init__()
        self.vision_model = Resnet(vision_config)
        # the layer that takes the image embedding with dimension (N, w' * h', image_dim) and converts it to (N, seq_len, text_dim)
        self.visual_proj = nn.Linear(vision_config.output_embed, language_config.embed_dim)
        self.language_model = GPT(language_config)
        self.language_config = language_config

    def forward(self, image, starting_text, labels=None):
        image = self.vision_model(image).transpose(1, 2)            
        image_embeddings = self.visual_proj(image)[:, :self.language_config.seq_len]       # We only want up to the sequence length after the projection for every single batch (which is the point of the first colon)
        x, loss = self.language_model(image_embeddings, starting_text, labels)            
        return x, loss

    # Don't worry about this. This basically just uses the model to generate tokens we would later decode to view the output in english. This is what we would use for inferencing.
    @torch.no_grad()
    def generate(self, image, starting_text, max_new_tokens, temperature=1, do_sample=True, top_k=None):
        print(image.shape)
        image = self.vision_model(image).transpose(1, 2)
        print(image.shape)
        image_embed = self.visual_proj(image)[:, :self.language_config.seq_len]
        print(image_embed)
        x, new = self.language_model.generate(image_embed, starting_text, max_new_tokens, temperature, do_sample, top_k)
        return x, new
