import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import LlamaForCausalLM
from tqdm import tqdm
import yaml
import logging
import time
from transformers import AutoTokenizer
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize the text
        tokens = tokenizer.encode(text)
        
        # Set maximum sequence length
        self.max_length = min(8192, seq_length)  # Cap at 8192 tokens
        self.seq_length = self.max_length
        
        # Create chunks of max_length
        self.chunks = []
        for i in range(0, len(tokens) - self.max_length + 1, self.max_length):
            chunk = tokens[i:i + self.max_length]
            if len(chunk) == self.max_length:  # Only keep complete chunks
                self.chunks.append(chunk)
        
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        
        # Create input and target sequences
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        
        # Ensure sequences are of correct length
        if input_ids.size(0) < self.seq_length - 1:
            padding = torch.full((self.seq_length - 1 - input_ids.size(0),), 
                               self.tokenizer.pad_token_id, 
                               dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
            labels = torch.cat([labels, padding])
            
        return input_ids, labels

def save_checkpoint(model, optimizer, scheduler, epoch, loss, config, checkpoint_dir):
    """Save model checkpoint with all training state"""
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'config': config
    } 
    
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Save as latest checkpoint
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    torch.save(checkpoint, latest_path)

def load_checkpoint(model, optimizer, scheduler, checkpoint_dir):
    """Load latest checkpoint if it exists"""
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    
    if os.path.exists(latest_path):
        logging.info(f"Loading checkpoint: {latest_path}")
        checkpoint = torch.load(latest_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['loss'], checkpoint['config']
    
    return 0, float('inf'), None

def get_device():
    """Determine the best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def evaluate_model(model, tokenizer, prompt="Hello, how are you?", max_length=50, device="cpu"):
    """Evaluate model by generating text from a prompt"""
    model.eval()
    with torch.no_grad():
        # Tokenize input prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Generate response
        outputs = model.generate(input_ids, max_length=max_length)
        
        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logging.info("\nModel Evaluation:")
        logging.info(f"Prompt: {prompt}")
        logging.info(f"Generated: {generated_text}\n")
        
        return generated_text

def print_model_summary(model, config):
    """Print detailed model summary"""
    # Calculate input shape based on config
    batch_size = config['tokens']['micro_batch_size']
    seq_length = config['tokens']['sequence_length']
    
    # Generate model summary
    # model_summary = summary(
    #     model,
    #     input_size=(batch_size, seq_length),
    #     col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
    #     depth=4,
    #     device="cpu"
    # )
    
    logging.info(f"Model Architecture:")
    logging.info(f"Model: {model}")
    logging.info(f"Hidden Size: {config['model']['model_config']['hidden_size']}")
    logging.info(f"Num Layers: {config['model']['model_config']['num_hidden_layers']}")
    logging.info(f"Num Attention Heads: {config['model']['model_config']['num_attention_heads']}")
    logging.info(f"Num KV Heads: {config['model']['model_config'].get('num_key_value_heads', config['model']['model_config']['num_attention_heads'])}")
    logging.info(f"Intermediate Size: {config['model']['model_config']['intermediate_size']}")
    logging.info(f"Vocabulary Size: {config['model']['model_config']['vocab_size']}\n")

def load_config():
    with open('SmolLM2-135.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Override model config to match exactly 135M parameters
    model_config = config['model']['model_config']
    model_config.update({
        'hidden_size': 552,           # Must be divisible by num_attention_heads
        'intermediate_size': 1536,    # Keep MLP intermediate size
        'num_attention_heads': 12,    # Changed to ensure proper division
        'num_key_value_heads': 4,     # 1/3 of attention heads
        'num_hidden_layers': 24,      # Keep number of layers
        'vocab_size': 49152,         # Keep vocab size
        'max_position_embeddings': 512,
        'rms_norm_eps': 1e-5,
        'hidden_act': 'silu'
    })
    
    return config

def train():

    config = load_config()
    
    
    # Set device
    device = get_device()
    device = "cpu"
    logging.info(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    
    # Create dataset and dataloader
    dataset = TextDataset('input.txt', tokenizer,  512)
                        #   config['tokens']['sequence_length'])
    dataloader = DataLoader(
        dataset,
        batch_size=config['tokens']['micro_batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda")
    )
    
    # Initialize model
    model = LlamaForCausalLM(config).to(device)
    
    # Print model summary
    print_model_summary(model, config)
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['learning_rate_scheduler']['learning_rate'],
        weight_decay=config['optimizer']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['tokens']['train_steps'],
        eta_min=config['optimizer']['learning_rate_scheduler'].get('min_decay_lr', 0)
    )
    
    # Load checkpoint if exists
    checkpoint_dir = Path(config['checkpoints']['checkpoints_path'])
    start_epoch, best_loss, loaded_config = load_checkpoint(model, optimizer, scheduler, checkpoint_dir)
    
    # Training loop
    num_epochs = 5000
    if(start_epoch != 0): num_epochs = start_epoch + 50
    
    checkpoint_interval = 100  # Save every 100 epochs
    
    # Main progress bar for epochs
    epoch_pbar = tqdm(range(start_epoch, num_epochs), 
                     desc="Training Progress",
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| [{n_fmt}/{total_fmt}] {postfix}',
                     position=0)
    
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0
        epoch_start_time = time.time()
        
        # Inner progress bar for batches
        batch_pbar = tqdm(enumerate(dataloader), 
                         total=len(dataloader),
                         desc=f"Batch Progress",
                         leave=False,
                         position=1)
        
        for batch_idx, (input_ids, labels) in batch_pbar:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids)
            loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['clip_grad'])
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            
            # Update batch progress bar
            batch_pbar.set_postfix({
                'batch_loss': f"{loss.item():.4f}"
            })
        
        # Close batch progress bar
        batch_pbar.close()
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - epoch_start_time
        
        # Update epoch progress bar
        epoch_pbar.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        epoch_pbar.set_postfix({
            'loss': f"{avg_epoch_loss:.4f}",
            'time': f"{epoch_time:.2f}s",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })
        
        # Save checkpoint every checkpoint_interval epochs
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler,
                epoch + 1, avg_epoch_loss,
                config, checkpoint_dir
            )
            
            # Log training summary
            logging.info(f"\nCheckpoint Summary (Epoch {epoch+1}):")
            logging.info(f"Average Loss: {avg_epoch_loss:.4f}")
            logging.info(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
            logging.info(f"Training Time: {epoch_time:.2f}s")
            
            # Run model evaluation
            evaluate_model(model, tokenizer, device=device)
        
        # Early stopping check
        if avg_epoch_loss < config['training']['target_loss']:
            logging.info(f"Target loss achieved at epoch {epoch+1}. Stopping training.")
            break
    
    # Save final model and evaluate
    save_checkpoint(
        model, optimizer, scheduler,
        epoch + 1, avg_epoch_loss,
        config, checkpoint_dir
    )
    evaluate_model(model, tokenizer, device=device)
    logging.info("Training completed!")

if __name__ == "__main__":
    train()
