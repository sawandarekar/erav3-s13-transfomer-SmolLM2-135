import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import LlamaForCausalLM
from tqdm import tqdm
from torchsummary import summary
import yaml
import logging
import time
from transformers import AutoTokenizer
import lorem
from torch.cuda.amp import GradScaler, autocast

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length):
        with open(file_path, 'r') as f:
            text = f.read()
        self.tokens = tokenizer.encode(text)
        self.seq_length = seq_length
        self.max_length = tokenizer.model_max_length
        # if len(self.tokens) > self.max_length:
        #     raise ValueError(f"Token indices sequence length is longer than the specified maximum sequence length for this model ({len(self.tokens)} > {self.max_length}).")

    def __len__(self):
        return len(self.tokens) // self.seq_length

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        return torch.tensor(self.tokens[start_idx:end_idx], dtype=torch.long)

def save_checkpoint(model, optimizer, scheduler, epoch, config):
    checkpoint_path = f"{config['checkpoints']['checkpoints_path']}/checkpoint_{epoch+1}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)
    logging.info(f"Checkpoint saved at epoch {epoch+1}")

def load_checkpoint(model, optimizer, scheduler, config):
    checkpoint_path = config['checkpoints']['resume_checkpoint_path']
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logging.info(f"Resumed training from checkpoint at epoch {checkpoint['epoch'] + 1}")
        return checkpoint['epoch'] + 1
    return 0

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def generate_sample_text(model, tokenizer, prompt_text, max_length=50):
    model.eval()
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

def train():
    # Load configuration from YAML
    with open('SmolLM2-135.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Set device
    device = get_device()
    device = "cpu"
    logging.info(f"Using device: {device}")

    # Load tokenizer before any multiprocessing
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    # Adjust model configuration to ensure 135 million parameters
    config['model']['hidden_size'] = 768  # Example adjustment
    config['model']['num_layers'] = 12    # Example adjustment
    config['model']['num_heads'] = 12     # Example adjustment
    if 'intermediate_size' not in config['model']:
        config['model']['intermediate_size'] = 1536  # Example adjustment

    # Initialize model
    model = LlamaForCausalLM(config['model']).to(device)
    scaler = GradScaler()  # Initialize GradScaler for mixed precision training
    num_parameters = sum(p.numel() for p in model.parameters())
    logging.info(f"Model initialized with {num_parameters} parameters")

    # Adjust batch size and sequence length for MPS
    if device == "mps":
        config['tokens']['micro_batch_size'] = min(config['tokens']['micro_batch_size'], 4)
        config['tokens']['sequence_length'] = min(config['tokens']['sequence_length'], 128)

    # Ensure sequence length does not exceed model's maximum sequence length
    config['tokens']['sequence_length'] = min(config['tokens']['sequence_length'], tokenizer.model_max_length)

    # Load dataset
    dataset = TextDataset('input.txt', tokenizer, config['tokens']['sequence_length'])
    dataloader = DataLoader(dataset, batch_size=config['tokens']['micro_batch_size'], shuffle=True, num_workers=0)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['learning_rate_scheduler']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['optimizer']['learning_rate_scheduler']['lr_decay_steps'], gamma=0.1)

    max_epoch = 5000
    # Load checkpoint if available
    start_epoch = load_checkpoint(model, optimizer, scheduler, config)
    if(start_epoch != 0): max_epoch = start_epoch + 50

    # Training loop
    accumulation_steps = config['training'].get('accumulation_steps', 1)
    for epoch in range(start_epoch, max_epoch):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for i, batch in enumerate(progress_bar):
            batch = batch.to(device)
            optimizer.zero_grad()
            if device in ["cuda", "mps"]:
                with autocast(device_type=device):  # Use autocast for mixed precision training
                    outputs = model(batch)
                    labels = batch[:, 1:].contiguous().view(-1)
                    logits = outputs[:, :-1].contiguous().view(-1, outputs.size(-1))
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    loss = loss / accumulation_steps  # Normalize loss by accumulation steps
                scaler.scale(loss).backward()  # Scale loss for mixed precision

                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)  # Unscale gradients and step optimizer
                    scaler.update()  # Update the scale for next iteration
                    optimizer.zero_grad()
            else:
                outputs = model(batch)
                labels = batch[:, 1:].contiguous().view(-1)
                logits = outputs[:, :-1].contiguous().view(-1, outputs.size(-1))
                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss = loss / accumulation_steps  # Normalize loss by accumulation steps
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += loss.item() * accumulation_steps  # Accumulate loss
            progress_bar.set_postfix(loss=loss.item() * accumulation_steps)

        scheduler.step()
        logging.info(f"Epoch {epoch+1} completed in {time.time() - start_time:.2f}s with loss {epoch_loss / len(dataloader):.4f}")

        if (epoch + 1) % config['checkpoints']['checkpoint_interval'] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, config)
            
            # Generate and print sample text
            text = lorem.sentence()
            sample_text = generate_sample_text(model, tokenizer, text)
            logging.info(f"Sample text: {text}")
            logging.info(f"Sample generated text: {sample_text}")

        if epoch_loss / len(dataloader) <= config['training']['target_loss']:
            logging.info("Target loss achieved. Stopping training.")
            break

if __name__ == "__main__":
    train()
