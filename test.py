import torch
import numpy as np

from config import parameters
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model_load import tokenizer
import random
import os
from pathlib import Path
import sys
from evaluate import load
from data_load import getDataSet

args = parameters
device = args.device
batch_size = args.batch_size

def directory_exists(filepath):
    if Path(filepath).is_dir():
        print("Results folder already exists, please rename")
        sys.exit(1)

''' Print loss graph'''

def plot_losses(training_loss_data, validation_loss_data, precision_bc, recall_bc, f1_bc, info, filepath):
    
    ''' Print Training/Validation Loss Graph '''
    tld = training_loss_data
    vld = validation_loss_data
    num_epoches = tld.shape[0]
    # l = np.mean(losses, axis=1)

    plt.subplot(1, 1, 1)
    plt.plot(range(num_epoches), tld, color='blue', label=f"Training Loss: {training_loss_data[-1]:.3f}", marker='o', alpha=0.5, ms=4) # print training loss data
    plt.plot(range(num_epoches), vld, color='orange', label=f"Validation Loss: {validation_loss_data[-1]:.3f}", marker='o', alpha=0.5, ms=4) # print validation loss data
    plt.title(f"Loss - {info}")
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.gcf().set_size_inches(6, 4)

    loss_graph_path = f"{filepath}/Loss_Graph.pdf"
    plt.savefig(loss_graph_path, bbox_inches='tight')
    print('Saved Loss Graph to %s' % (loss_graph_path))
    plt.clf()

    ''' Print Bertscore graphs according to category (3) '''

    # Print Precision Graph
    pbcd = precision_bc
    plt.subplot(1, 1, 1)
    plt.plot(range(num_epoches), pbcd, color='blue', label=f"Precision: {precision_bc[-1]:.3f}", marker='o', alpha=0.5, ms=4) # print precision data
    plt.title(f"Precision - {info}")
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.gcf().set_size_inches(6, 4)

    precision_path = f"{filepath}/Precision.pdf"
    plt.savefig(precision_path, bbox_inches='tight')
    print('Saved Precision Graph to %s' % (precision_path))
    plt.clf()

    # Print Recall Graph
    rbcd = recall_bc
    plt.subplot(1, 1, 1)
    plt.plot(range(num_epoches), rbcd, color='blue', label=f"Precision: {recall_bc[-1]:.3f}", marker='o', alpha=0.5, ms=4) # print precision data
    plt.title(f"Recall - {info}")
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.gcf().set_size_inches(6, 4)

    recall_path = f"{filepath}/Recall.pdf"
    plt.savefig(recall_path, bbox_inches='tight')
    print('Saved Recall Graph to %s' % (recall_path))
    plt.clf()

    # Print F1 Score Graph
    f1bcd = f1_bc
    plt.subplot(1, 1, 1)
    plt.plot(range(num_epoches), f1bcd, color='blue', label=f"Precision: {f1_bc[-1]:.3f}", marker='o', alpha=0.5, ms=4) # print precision data
    plt.title(f"F1 Scores - {info}")
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.gcf().set_size_inches(6, 4)

    f1_path = f"{filepath}/F1_Scores.pdf"
    plt.savefig(f1_path, bbox_inches='tight')
    print('Saved F1 Graph to %s' % (f1_path))
    plt.clf()

def tensor_to_image(tensor):
    # Convert the tensor to a PIL Image
    return transforms.ToPILImage()(tensor)

def generate_caption(image, model, tokenizer, max_length=50, temperature=1.0):

    caption = [tokenizer.bos_token_id]

    model.eval()

    with torch.no_grad():
        for i in range(max_length):
            input_ids = torch.LongTensor(caption).unsqueeze(0)
            input_ids = input_ids.to(device)
            outputs = model(image.unsqueeze(0), input_ids, attention_mask=None)

            logits = outputs.logits[:, -1, :] / temperature
            predicted_id = logits.argmax(1).item()
            caption.append(predicted_id)

            if predicted_id == tokenizer.eos_token_id and i > 1:
                break
    generated_caption = tokenizer.decode(caption, skip_special_tokens=True)
    return generated_caption

def show_results(model, val_dataloader, training_loss_data, validation_loss_data, precision_bc, recall_bc, f1_bc, info, filepath):

    ''' Save loss graph in results folder '''
    results_folder_path = f"Results/{filepath}"
    os.mkdir(results_folder_path) # Create result set in results folder
    plot_losses(training_loss_data, validation_loss_data, precision_bc, recall_bc, f1_bc, info, results_folder_path)

    ''' Generate 5 Image Captions per result'''
    image_captions_path = f"{results_folder_path}/ImageCaptions"
    os.mkdir(image_captions_path) # Make image directory
    for i in range(1, 6):

        # Retrieve Random Image
        random_idx = random.randint(0, len(val_dataloader) - 1)
        images, _, _ = next(iter(val_dataloader))
        image = images[random_idx].to(device)
        img = tensor_to_image(image)

        # Generate Caption
        caption = generate_caption(image, model, tokenizer)

        # Create figure
        ax = plt.figure().add_subplot(111)

        # Add image
        ax.imshow(img)

        # Add generated caption
        plt.title(caption, loc='center', wrap=True)

        # Save figure on results set folder
        img_cap_path = f"{image_captions_path}/img_cap_{i}.pdf"
        plt.savefig(img_cap_path)
        print('Saved Generated Caption to %s' % (img_cap_path))
        plt.clf()



''' No training case '''

def plot_bertscore(precision_bc, recall_bc, f1_bc, info, filepath):

    ''' Print Bertscore graphs according to category (3) '''

    num_epoches = precision_bc.shape[0]

    # Print Precision Graph
    pbcd = precision_bc
    plt.subplot(1, 1, 1)
    plt.plot(range(num_epoches), pbcd, color='blue', label=f"Precision: {precision_bc[-1]:.3f}", marker='o', alpha=0.5, ms=4) # print precision data
    plt.title(f"Precision - {info}")
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.gcf().set_size_inches(6, 4)

    precision_path = f"{filepath}/Precision.pdf"
    plt.savefig(precision_path, bbox_inches='tight')
    print('Saved Precision Graph to %s' % (precision_path))
    plt.clf()

    # Print Recall Graph
    rbcd = recall_bc
    plt.subplot(1, 1, 1)
    plt.plot(range(num_epoches), rbcd, color='blue', label=f"Precision: {recall_bc[-1]:.3f}", marker='o', alpha=0.5, ms=4) # print precision data
    plt.title(f"Recall - {info}")
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.gcf().set_size_inches(6, 4)

    recall_path = f"{filepath}/Recall.pdf"
    plt.savefig(recall_path, bbox_inches='tight')
    print('Saved Recall Graph to %s' % (recall_path))
    plt.clf()

    # Print F1 Score Graph
    f1bcd = f1_bc
    plt.subplot(1, 1, 1)
    plt.plot(range(num_epoches), f1bcd, color='blue', label=f"Precision: {f1_bc[-1]:.3f}", marker='o', alpha=0.5, ms=4) # print precision data
    plt.title(f"F1 Scores - {info}")
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.gcf().set_size_inches(6, 4)

    f1_path = f"{filepath}/F1_Scores.pdf"
    plt.savefig(f1_path, bbox_inches='tight')
    print('Saved F1 Graph to %s' % (f1_path))
    plt.clf()


def show_results_no_training(model, dataset, info, filepath):

    ''' Create and split dataset '''
    _, val_dataloader = getDataSet(batch_size, dataset)

    ''' Make BertScore Graphs for each category (3) '''
    bertscore = load("bertscore")
    precision_bc = np.zeros((5), dtype=np.float32)
    recall_bc = np.zeros((5), dtype=np.float32)
    f1_bc = np.zeros((5), dtype=np.float32)

    ''' Generate 5 Image Captions per result'''
    results_folder_path = f"Results/{filepath}"
    os.mkdir(results_folder_path) # Create result set in results folder
    image_captions_path = f"{results_folder_path}/ImageCaptions"
    os.mkdir(image_captions_path) # Make image directory
    for i in range(1, 6):

        # Retrieve Random Image
        random_idx = random.randint(0, len(val_dataloader) - 1)
        images, captions, _ = next(iter(val_dataloader))
        image = images[random_idx].to(device)
        img = tensor_to_image(image)

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
        precision_bc[i-1] = precision
        recall_bc[i-1] = recall
        f1_bc[i-1] = f1

        # Create figure
        ax = plt.figure().add_subplot(111)

        # Add image
        ax.imshow(img)

        # Add generated caption
        plt.title(prediected_caption, loc='center', wrap=True)

        # Save figure on results set folder
        img_cap_path = f"{image_captions_path}/img_cap_{i}.pdf"
        plt.savefig(img_cap_path)
        print('Saved Generated Caption to %s' % (img_cap_path))
        plt.clf()

    # Plot the bertscores
    plot_bertscore(precision_bc, recall_bc, f1_bc, info, results_folder_path)
