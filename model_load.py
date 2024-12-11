from transformers import GPT2LMHeadModel, TFGPT2LMHeadModel, FlaxGPT2LMHeadModel, GPT2Config
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from transformers import GPT2Tokenizer

''' Load Tokenizer '''
tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # Create a blank Byte-Pair Encoding tokenizer

''' Load ResNet '''
def getResNet(version, pretrained):

    if version == "resnet18":
        return resnet18(pretrained=pretrained)
    elif version == "resnet34":
        return resnet34(pretrained=pretrained)
    elif version == "resnet50":
        return resnet50(pretrained=pretrained)
    elif version == "resnet101":
        return resnet101(pretrained=pretrained)
    elif version == "resnet152":
        return resnet152(pretrained=pretrained)
    else:
        raise ValueError(f"{version} - is not a valid resnet model")

''' Load GPT2 model '''
def getGPT2LMHeadModel(version, pretrained):

    if version == "GPT2LMHeadModel":
        if pretrained:
            return GPT2LMHeadModel.from_pretrained('gpt2')
        else: 
            return GPT2LMHeadModel(GPT2Config())
    elif version == "TFGPT2LMHeadModel":
        if pretrained:
            return TFGPT2LMHeadModel.from_pretrained('gpt2')
        else: 
            return TFGPT2LMHeadModel(GPT2Config())
    elif version == "FlaxGPT2LMHeadModel":
        if pretrained:
            return FlaxGPT2LMHeadModel.from_pretrained('gpt2')
        else: 
            return FlaxGPT2LMHeadModel(GPT2Config())
    else:
        raise ValueError(f"{version} - is not a valid GPT2LM model")