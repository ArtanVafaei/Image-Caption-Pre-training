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

resnet = resent_config.resnet_config
gpt2 = gpt2_config.gpt2_config
args = trainer_config.trainer_args

if args.log:
    import wandb
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in args.device else 'cpu' # for later use in torch.autocast

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
auto = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

model = VLM(resnet, gpt2).to(device=device_type)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

print(f"The Model has {sum([p.numel() for p in model.parameters()]) / 1e6:.2f} Million Parameters")

assert args.train_dataset_size + args.test_dataset_size <= 10000, f"The dataset only contains 10000 samples. The total number of training and testing data samples should not exceed 10000. Please change train and test data size in the trainer_config file."

train_dataset = ImageTextDataset(args.train_dataset_size)
test_dataset = ImageTextDataset(args.test_dataset_size)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

starting_iter = 0

if args.resume:
    assert os.path.exists(args.output_dir)
    checkpoint = torch.load(os.path.join(args.output_dir, 'ckpt.pt'), map_location=device_type)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    starting_iter = checkpoint['iter_num']
    checkpoint = None

def avg(x: list):
    return sum(x) / len(x)

starting_text = torch.zeros((1, gpt2.seq_len), dtype=torch.long).to(device_type)
# torch.autograd.detect_anomaly(True)

# len(dataset.validation)
def eval():
    val_loss = []
    model.eval()
    with torch.no_grad():
        for (image, text) in test_dataloader:
            image = image.to(device=device_type)
            text = text.to(device=device_type)
            with auto:
                logits, loss = model(image, starting_text, text)
            val_loss.append(loss)
            avg_val_loss = avg(val_loss) if len(val_loss) != 0 else 0
    print(f"Validation Loss: {avg_val_loss:.4f}", end='\t')
    return avg_val_loss
            
import time

start = time.time()

if args.save_checkpoint:
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        
tokenizer = tiktoken.get_encoding("gpt2") 
decode = lambda l: tokenizer.decode([token for token in l.tolist() if token != 0])

# len(dataset)
for i in range(args.num_iterations):
    train_loss = []

    if i % 10 == 0:
        # after each ith iteration, we would like to compare the model's output with the actual output
        rand_num = np.random.randint(0, len(test_dataset), size=1)[0]
        _, output = model.generate(test_dataset[rand_num][0].to(device_type).unsqueeze(0), starting_text, max_new_tokens=100)

        print("Predicted Captions:\n")
        print([tokenizer.decode(output.tolist())]) 
        print("\nActual Captions:\n")
        print([tokenizer.decode(test_dataset[rand_num][1].tolist())])

    model.train()

    for batch_idx, (image, text) in enumerate(train_dataloader):
        # moving the image and text to cuda if its available (it is already in cpu so if cuda is not available, nothing will happen)
        image = image.to(device=device_type)
        text = text.to(device=device_type)

        # enables mixed precision if fp16 is enabled
        with auto:
            logits, loss = model(image, starting_text, labels=text)
            loss = loss / args.gradient_accumulation_steps      # scaling loss to account for gradient accumulation
        
        train_loss.append(loss.item() * args.gradient_accumulation_steps)
        scaler.scale(loss).backward()

        if (batch_idx) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip) 
            optimizer.step() 
            scaler.update()
            optimizer.zero_grad()
    
    # evaluation, logging, and printing
    print(f"Epoch Number: {i + 1}", end='\t')
    print(f"Training Loss: {avg(train_loss):.4f}", end='\t')
    val_loss = eval()
    t = time.time() - start
    print(f"Current Time: {t:.4f} seconds\t {(t / 60):.4f} minutes\t {(t / 3600):4f} hours")

    if args.log:
        wandb.log({
            "iter": i,
            "time": t / 3600,
            "train/loss": avg(train_loss),
            "val/loss": val_loss,
        })

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
