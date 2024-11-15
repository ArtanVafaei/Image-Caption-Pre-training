from model import VLM
from config import resent_config, gpt2_config, trainer_config
from data import ImageTextDataset
from torch.utils.data import DataLoader
import tiktoken
import torch
import numpy as np
from contextlib import nullcontext
import torchvision.transforms as transforms
from PIL import Image
import os
import time
import sys

'''
main.py: training model
NOTE: If dataset.npz does not exist, run prep_data.py before running main.py
'''

''' Load hyperparameters and load GPU integrations '''
# Loading hyperparemeters
resnet = resent_config.resnet_config
gpt2 = gpt2_config.gpt2_config
args = trainer_config.trainer_args

# Allow GPU to run processes
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in args.device else 'cpu' # for later use in torch.autocast

# GPU: bfloat16 (supported), CPU: float16
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype] 
auto = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

''' Integrate Bertscore to display accuracy '''
from evaluate import load
bertscore = load("bertscore")

''' Initialize our ResNet-GPT model and optimizer'''
model = VLM(resnet, gpt2).to(device=device_type) # Initialize Vision-Language Model
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # Initialize Optimizer

# Print out parameter count
print(f"The Model has {sum([p.numel() for p in model.parameters()]) / 1e6:.2f} Million Parameters")

# Datasize limit
assert args.train_dataset_size + args.test_dataset_size <= 10000, f"The dataset only contains 10000 samples. The total number of training and testing data samples should not exceed 10000. Please change train and test data size in the trainer_config file."

''' Creating dataset and dataloader [from prep_data.py] '''
# Train/test dataset
train_dataset = ImageTextDataset(args.train_dataset_size, train=True)
test_dataset = ImageTextDataset(args.test_dataset_size, train=True)

# Train/test dataloader
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

starting_iter = 0

''' Training interrupt/fail backup for model parameters'''
if args.resume:
    assert os.path.exists(args.output_dir)
    checkpoint = torch.load(os.path.join(args.output_dir, 'ckpt.pt'), map_location=device_type)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    starting_iter = checkpoint['iter_num']
    checkpoint = None

''' Evaluation for each image '''
def avg(x: list):
    return sum(x) / len(x)

# Tensor concatenates w/ image embds, for VQA integration
starting_text = torch.zeros((1, gpt2.seq_len), dtype=torch.long).to(device_type)

# Evaluation
def eval():
    val_loss = []
    model.eval() # Evaluation mode
    with torch.no_grad():
        for (image, text) in test_dataloader:
            image = image.to(device=device_type)
            text = text.to(device=device_type)
            with auto:
                _, loss = model(image, starting_text, text) # Forward pass
            val_loss.append(loss)
            avg_val_loss = avg(val_loss) if len(val_loss) != 0 else 0
    print(f"Validation: {avg_val_loss:.4f}", end='\n')
    return avg_val_loss

''' Save checkpoint to resume training if enabled '''
if args.save_checkpoint:
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

''' Create Decoder '''
tokenizer = tiktoken.get_encoding("gpt2")  # Tokenize
decode = lambda l: tokenizer.decode([token for token in l.tolist() if token != 0]) # Make English

''' Iterate through all Epochs '''
start = time.time() # Create a start time
for i in range(args.num_iterations):
    train_loss = []

    ''' Every 10th Epoch, compare predicted and actual output'''
    if i % 10 == 0:
        rand_num = np.random.randint(0, len(test_dataset), size=1)[0] # Get random index
        _, output = model.generate(test_dataset[rand_num][0].to(device_type).unsqueeze(0), starting_text, max_new_tokens=100) # Generate predicted captions

        # Print out results and Bertscore
        prediction = [tokenizer.decode(output.tolist())]
        actual = [tokenizer.decode(test_dataset[rand_num][1].tolist())]
        results = bertscore.compute(predictions=prediction, references=actual, model_type="distilbert-base-uncased")
        print(f"Predicted Captions:\n{prediction}\nActual Captions:\n{actual}")
        print(f"BertScore -\t   Precision: {results.get('precision')[0]:.4f}\tRecall: {results.get('recall')[0]:.4f}\tF1 Score: {results.get('f1')[0]:.4f}")

    model.train() # Activate training mode for model

    ''' Go through each batch in the dataloader'''
    for batch_idx, (image, text) in enumerate(train_dataloader):

        # Transfer image/text to GPU process if possible
        image = image.to(device=device_type)
        text = text.to(device=device_type)

        # Enables mixed precision if fp16 is enabled
        with auto:
            logits, loss = model(image, starting_text, labels=text)
            loss = loss / args.gradient_accumulation_steps      # scaling loss to account for gradient accumulation
        
        train_loss.append(loss.item() * args.gradient_accumulation_steps)
        scaler.scale(loss).backward() # backward pass

        # Weights will be updated after "gradient_accumulation_steps" times
        if (batch_idx) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip) 
            optimizer.step()  # Weight update
            scaler.update()
            optimizer.zero_grad() # Zero out gradients
    
    ''' Evaluation, Logging, and Printing'''
    t = time.time() - start
    print(f"{int(t/3600)}:{int(t/60)}:{int(t%60)} -\t   Epoch: {i + 1}   Loss - Training: {avg(train_loss):.4f}   ", end="")
    val_loss = eval()
    # print(f"Current Time: {t:.4f} seconds\t {(t / 60):.4f} minutes\t {(t / 3600):4f} hours")

    ''' Save Checkpoint if enabled'''
    if args.save_checkpoint:
        checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': gpt2,
                    'iter_num': i,
                    'best_val_loss': val_loss,
                    'config': args,
                }
        print(f"saving checkpoint to {args.output_dir}")
        torch.save(checkpoint, os.path.join(args.output_dir, 'ckpt.pt'))
        checkpoint = None
        

checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': gpt2,
            'iter_num': i,
            'best_val_loss': val_loss,
            'config': args,
        }
print(f"saving checkpoint to {args.output_dir}") # if args.save_checkpoint == False, ignore
torch.save(checkpoint, os.path.join(args.output_dir, 'final_model.pt'))
