import torch.nn as nn

class ImageCaptioningModel(nn.Module):
    def __init__(self, resnet, resnet_model, gpt2):
        super(ImageCaptioningModel, self).__init__()
        self.resnet = resnet
        self.gpt2 = gpt2

        out_feature = 0
        if resnet_model in ["resnet50", "resnet101", "resnet152"]:
            out_feature = 2048
        else:
            out_feature = 512 # resnet18, resnet34
        
        self.proj = nn.Linear(out_feature, gpt2.config.hidden_size)

    def forward(self, images, input_ids, attention_mask=None):
        img_features = self.resnet(images)
        img_features = img_features.mean([2,3])
        img_features = self.proj(img_features)

        input_embeddings = self.gpt2.transformer.wte(input_ids)
        combined_embeddings =  input_embeddings + img_features.unsqueeze(1)

        outputs = self.gpt2(inputs_embeds=combined_embeddings, attention_mask=attention_mask, labels=input_ids)
        return outputs