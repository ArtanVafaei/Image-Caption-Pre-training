from dataclasses import dataclass
from config.gpt2_config import gpt2_config
import torch

@dataclass
class trainer_args:
    train_dataset_size = 200
    test_dataset_size = 20   
    batch_size = 4
    num_iterations = 100        # epochs --> how many times the model learns from each sample in the data
    gradient_accumulation_steps = 4         # After how many batches will the model's weights be updated
    grad_clip = 1.0
    lr = 4e-4

    image_size = (224, 224)
    seq_len = gpt2_config.seq_len

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    
    log = False
    wandb_project = "idk"
    wandb_run_name = "image_caption"
    output_dir = '../out'
    save_checkpoint = False

    resume = False


