import torch.nn as nn
from train import train
from test import show_results, show_results_no_training, directory_exists
from model_load import getResNet, getGPT2LMHeadModel
from data_load import getDataSetName
from model import ImageCaptioningModel
import sys
from config import parameters

args = parameters
device = args.device
learning_rate = args.learning_rate

def exampleTest():
    # Set model versions, pretrain condition, and dataset
    resnet_model = "resnet50"
    gpt2lmhead_model = "GPT2LMHeadModel"
    pretrained = True
    pretrained_str = "Pretrained" if pretrained else "Trained"
    dataset = "dataset1"
    filepath = "Test_Results"
    info = f"{resnet_model}, {gpt2lmhead_model}, {getDataSetName(dataset)}, {pretrained_str}" # Set results name

    # Test if results folder already exists else end the program
    directory_exists(f"Results/{filepath}")
    print(f"Testing: {info}") # Passes test

    # Train the model with the dataset
    model, test_dataloader, training_loss_data, validation_loss_data, precision_bc, recall_bc, f1_bc = train(
        resnet_model=resnet_model, 
        gpt2lmhead_model=gpt2lmhead_model, 
        pretrained=pretrained, 
        dataset=dataset
    )
    
    # Saves loss graph, 5 image captions
    show_results(
        model=model, 
        val_dataloader=test_dataloader, 
        training_loss_data=training_loss_data, 
        validation_loss_data=validation_loss_data, 
        precision_bc=precision_bc,
        recall_bc=recall_bc,
        f1_bc=f1_bc,
        info=info, 
        filepath=filepath
    ) 

''' Resnet50, GPTLMHead, Training, Pretrained Weights '''
def test1():

    # Set model versions, pretrain condition, and dataset
    resnet_model = "resnet50"
    gpt2lmhead_model = "GPT2LMHeadModel"
    pretrained = True
    pretrained_str = "Pretrained" if pretrained else "Trained"
    dataset = "dataset1"
    filepath = "Test_1_Resnet50_GPTLMHead_Training_Pretrained Weights"
    info = f"{resnet_model}, {gpt2lmhead_model}, {getDataSetName(dataset)}, {pretrained_str}" # Set results name

    # Test if results folder already exists else end the program
    directory_exists(f"Results/{filepath}")
    print(f"Testing: {info}") # Passes test

    # Train the model with the dataset
    model, test_dataloader, training_loss_data, validation_loss_data, precision_bc, recall_bc, f1_bc = train(
        resnet_model=resnet_model, 
        gpt2lmhead_model=gpt2lmhead_model, 
        pretrained=pretrained, 
        dataset=dataset
    )
    
    # Saves loss graph, 5 image captions
    show_results(
        model=model, 
        val_dataloader=test_dataloader, 
        training_loss_data=training_loss_data, 
        validation_loss_data=validation_loss_data, 
        precision_bc=precision_bc,
        recall_bc=recall_bc,
        f1_bc=f1_bc,
        info=info, 
        filepath=filepath
    ) 

''' Resnet50, GPTLMHead, No Training, Pretrained Weights '''
def test2():

    # Set model versions, pretrain condition, and dataset
    resnet_model = "resnet50"
    gpt2lmhead_model = "GPT2LMHeadModel"
    pretrained = True
    pretrained_str = "Pretrained" if pretrained else "Trained"
    dataset = "dataset1"
    filepath = "Test_2_Resnet50_GPTLMHead_No_Training_Pretrained Weights"
    info = f"{resnet_model}, {gpt2lmhead_model}, {getDataSetName(dataset)}, {pretrained_str}" # Set results name

    # Test if results folder already exists else end the program
    directory_exists(f"Results/{filepath}")
    print(f"Testing: {info}") # Passes test

    ''' Load in ResNet and GPT2 model '''
    resnet = getResNet(version=resnet_model, pretrained=pretrained).to(device)
    resnet = nn.Sequential(*list(resnet.children())[:-2]) # Remove avg pool and fc layer
    gpt2 = getGPT2LMHeadModel(version=gpt2lmhead_model, pretrained=pretrained).to(device)
    
    ''' Create Model with optimizer '''
    model = ImageCaptioningModel(resnet, resnet_model, gpt2).to(device)

    # Saves loss graph, 5 image captions
    show_results_no_training(
        model=model, 
        dataset=dataset,
        info=info, 
        filepath=filepath
    ) 

''' Resnet50, GPTLMHead, Training, No Pretrained Weights '''
def test3():
    # Set model versions, pretrain condition, and dataset
    resnet_model = "resnet50"
    gpt2lmhead_model = "GPT2LMHeadModel"
    pretrained = False
    pretrained_str = "Pretrained" if pretrained else "Trained"
    dataset = "dataset1"
    filepath = "Test_3_Resnet50_GPTLMHead_Training_No_Pretrained Weights"
    info = f"{resnet_model}, {gpt2lmhead_model}, {getDataSetName(dataset)}, {pretrained_str}" # Set results name

    # Test if results folder already exists else end the program
    directory_exists(f"Results/{filepath}")
    print(f"Testing: {info}") # Passes test

    # Train the model with the dataset
    model, test_dataloader, training_loss_data, validation_loss_data, precision_bc, recall_bc, f1_bc = train(
        resnet_model=resnet_model, 
        gpt2lmhead_model=gpt2lmhead_model, 
        pretrained=pretrained, 
        dataset=dataset
    )
    
    # Saves loss graph, 5 image captions
    show_results(
        model=model, 
        val_dataloader=test_dataloader, 
        training_loss_data=training_loss_data, 
        validation_loss_data=validation_loss_data, 
        precision_bc=precision_bc,
        recall_bc=recall_bc,
        f1_bc=f1_bc,
        info=info, 
        filepath=filepath
    ) 

''' Resnet34, GPTLMHead, Training, Pretrained Weights '''
def test4():
    # Set model versions, pretrain condition, and dataset
    resnet_model = "resnet34"
    gpt2lmhead_model = "GPT2LMHeadModel"
    pretrained = True
    pretrained_str = "Pretrained" if pretrained else "Trained"
    dataset = "dataset1"
    filepath = "Test_4_Resnet34_GPTLMHead_Training_Pretrained Weights"
    info = f"{resnet_model}, {gpt2lmhead_model}, {getDataSetName(dataset)}, {pretrained_str}" # Set results name

    # Test if results folder already exists else end the program
    directory_exists(f"Results/{filepath}")
    print(f"Testing: {info}") # Passes test

    # Train the model with the dataset
    model, test_dataloader, training_loss_data, validation_loss_data, precision_bc, recall_bc, f1_bc = train(
        resnet_model=resnet_model, 
        gpt2lmhead_model=gpt2lmhead_model, 
        pretrained=pretrained, 
        dataset=dataset
    )
    
    # Saves loss graph, 5 image captions
    show_results(
        model=model, 
        val_dataloader=test_dataloader, 
        training_loss_data=training_loss_data, 
        validation_loss_data=validation_loss_data, 
        precision_bc=precision_bc,
        recall_bc=recall_bc,
        f1_bc=f1_bc,
        info=info, 
        filepath=filepath
    ) 

''' Resnet34, GPTLMHead, No Training, Pretrained Weights '''
def test5():
    # Set model versions, pretrain condition, and dataset
    resnet_model = "resnet34"
    gpt2lmhead_model = "GPT2LMHeadModel"
    pretrained = True
    pretrained_str = "Pretrained" if pretrained else "Trained"
    dataset = "dataset1"
    filepath = "Test_5_Resnet34_GPTLMHead_No_Training_Pretrained Weights"
    info = f"{resnet_model}, {gpt2lmhead_model}, {getDataSetName(dataset)}, {pretrained_str}" # Set results name

    # Test if results folder already exists else end the program
    directory_exists(f"Results/{filepath}")
    print(f"Testing: {info}") # Passes test

    ''' Load in ResNet and GPT2 model '''
    resnet = getResNet(version=resnet_model, pretrained=pretrained).to(device)
    resnet = nn.Sequential(*list(resnet.children())[:-2]) # Remove avg pool and fc layer
    gpt2 = getGPT2LMHeadModel(version=gpt2lmhead_model, pretrained=pretrained).to(device)
    
    ''' Create Model with optimizer '''
    model = ImageCaptioningModel(resnet, resnet_model, gpt2).to(device)

    # Saves loss graph, 5 image captions
    show_results_no_training(
        model=model, 
        dataset=dataset,
        info=info, 
        filepath=filepath
    ) 

''' Resnet34, GPTLMHead, Training, No Pretrained Weights '''
def test6():
    # Set model versions, pretrain condition, and dataset
    resnet_model = "resnet34"
    gpt2lmhead_model = "GPT2LMHeadModel"
    pretrained = False
    pretrained_str = "Pretrained" if pretrained else "Trained"
    dataset = "dataset1"
    filepath = "Test_6_Resnet34_GPTLMHead_Training_No_Pretrained Weights"
    info = f"{resnet_model}, {gpt2lmhead_model}, {getDataSetName(dataset)}, {pretrained_str}" # Set results name

    # Test if results folder already exists else end the program
    directory_exists(f"Results/{filepath}")
    print(f"Testing: {info}") # Passes test

    # Train the model with the dataset
    model, test_dataloader, training_loss_data, validation_loss_data, precision_bc, recall_bc, f1_bc = train(
        resnet_model=resnet_model, 
        gpt2lmhead_model=gpt2lmhead_model, 
        pretrained=pretrained, 
        dataset=dataset
    )
    
    # Saves loss graph, 5 image captions
    show_results(
        model=model, 
        val_dataloader=test_dataloader, 
        training_loss_data=training_loss_data, 
        validation_loss_data=validation_loss_data, 
        precision_bc=precision_bc,
        recall_bc=recall_bc,
        f1_bc=f1_bc,
        info=info, 
        filepath=filepath
    ) 

def main():

    ''' 
    *** BEFORE RUNNING MAIN ***
    Run data_install.py to locally store the datasets for testing
    '''

    '''
    Valid train arguments - *** Case sensitive ***

    Resnet Versions:    resnet18, resnet34, resnet50, resnet101, resnet152
    GPT Versions:       GPT2LMHeadModel (PyTorch subclass), TFGPT2LMHeadModel (keras subclass), FlaxGPT2LMHeadModel (flax subclass)
    pretrained:         True, False
    dataset:            dataset1 (Flickr8kDataset)
    '''

    ''' Example training '''
    # exampleTest()

    ''' Research Paper Results '''

    ''' Test 1: Resnet50, GPTLMHead, Training, Pretrained Weights '''
    # test1()

    ''' Test 2: Resnet50, GPTLMHead, ** No Training **, Pretrained Weights '''
    # test2()

    ''' Test 3: Resnet50, GPTLMHead, Training, No Pretrained Weights '''
    test3()

    ''' Test 4: Resnet34, GPTLMHead, Training, Pretrained Weights '''
    # test4()

    ''' Test 5: Resnet34, GPTLMHead, ** No Training **, Pretrained Weights '''
    # test5()

    ''' Test 6: Resnet34, GPTLMHead, Training, No Pretrained Weights '''
    # test6()

if __name__=="__main__":
    main()