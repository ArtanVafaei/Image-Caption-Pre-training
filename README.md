# End-to-End Image-Captioning Training

## Implementation Details
This program takes a ResNet and GPT2 model and trains on a Flickr8K dataset to output a predicted image with a predicted caption

ResNet models are from PyTorch libraries  
- Pretrained ResNet weights are from training a ImageNet set

GPT2 models are from Hugging Face libraries  
- Pretrained GPT2 weights comes from training very large corpus of ~40 GB of text data  

## Running the program
1. **Run data_install.py** to locally download dataset for training 
2. **Edit main.py** for desired ResNet version, GPT2 version, dataset, pretrained mode
3. **Run main.py** to test the results

### Testing Categories
1. Image Captioning by training the model *without* pre-trained weights [Goal results]  
2. Image Captioning by training the model *with* pre-trained weights
- Use various resnets and GPT models for testing <ins> listed in main.py </ins>


### To do:
- [x] Put results in a folder instead of directly displaying after training
- [x] Implement BertScore and Results Graph

## Credits
Nicolas Deperrois 
- https://www.kaggle.com/code/ndeperrois/resnet-gpt2
- https://colab.research.google.com/drive/1SVd9oy4rmt6avoDbnS2aIddiLtf77wf7#scrollTo=KDc3szmqlyxb
- _This was a modification of the owner's code which includes scalable testing across different language models_
- _"This Notebook has been released under the Apache 2.0 open source license."_

TorchVision Models  
- https://github.com/pytorch/vision  

HuggingFace OpenAI Models
- https://huggingface.co/docs/transformers/en/model_doc/gpt2

## Reference Repositories
https://github.com/Michael-Tanjuakio/End-to-End-Image-Captioning-Training

## Owners of Repository
Michael Tanjuakio, Sowresh Mecheri-Senthil, Artan Vafaei, Anthony Nguyen
