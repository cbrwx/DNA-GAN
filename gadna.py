import os
import random
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set random seeds for reproducibility
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class Environment:
    temperature: float
    pH: float
    chemical_exposure: float
    radiation: float

    def validate(self) -> None:
        if not 0 <= self.temperature <= 100:
            raise ValueError("Temperature must be between 0 and 100")
        if not 0 <= self.pH <= 14:
            raise ValueError("pH must be between 0 and 14")
        if not 0 <= self.chemical_exposure <= 1:
            raise ValueError("Chemical exposure must be between 0 and 1")
        if not 0 <= self.radiation <= 1:
            raise ValueError("Radiation must be between 0 and 1")

@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    hidden_dim: int = 128
    clip_grad: float = 1.0
    patience: int = 10
    min_delta: float = 0.001

class DNATokenizer:
    def __init__(self):
        self.token2idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '<EOS>': 4, '<PAD>': 5}
        self.idx2token = {v: k for k, v in self.token2idx.items()}

    def encode(self, dna_sequence: str) -> List[int]:
        try:
            return [self.token2idx[token] for token in dna_sequence] + [self.token2idx['<EOS>']]
        except KeyError as e:
            raise ValueError(f"Invalid DNA sequence character: {e}")

    def decode(self, encoded_sequence: List[int]) -> str:
        return ''.join(self.idx2token[idx] for idx in encoded_sequence if idx not in 
                      {self.token2idx['<EOS>'], self.token2idx['<PAD>']})

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x)
        return self.norm(x + attn_output)

class Generator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, environment: Environment):
        super().__init__()
        self.environment = environment
        self.env_dim = 4
        
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.env_encoder = nn.Sequential(
            nn.Linear(self.env_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.attention = AttentionLayer(hidden_dim)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        env_input = torch.tensor(
            [self.environment.temperature, self.environment.pH,
             self.environment.chemical_exposure, self.environment.radiation],
            dtype=torch.float32
        ).repeat(batch_size, 1).to(x.device)
        
        embedded = self.embedding(x)
        env_encoded = self.env_encoder(env_input).unsqueeze(1).repeat(1, x.size(1), 1)
        
        combined = embedded + env_encoded
        lstm_out, _ = self.lstm(combined)
        attended = self.attention(lstm_out)
        
        return torch.softmax(self.output_layer(attended), dim=-1)

class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, environment: Environment):
        super().__init__()
        self.environment = environment
        self.env_dim = 4
        
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.env_encoder = nn.Sequential(
            nn.Linear(self.env_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.attention = AttentionLayer(hidden_dim)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._process(x)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return self._process(x)

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        env_input = torch.tensor(
            [self.environment.temperature, self.environment.pH,
             self.environment.chemical_exposure, self.environment.radiation],
            dtype=torch.float32
        ).repeat(batch_size, 1).to(x.device)
        
        embedded = self.embedding(x)
        env_encoded = self.env_encoder(env_input).unsqueeze(1).repeat(1, x.size(1), 1)
        
        combined = embedded + env_encoded
        lstm_out, _ = self.lstm(combined)
        attended = self.attention(lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attended, dim=1)
        return self.output_layer(pooled)

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

class ModelCheckpoint:
    def __init__(self, filepath: str, save_best_only: bool = True):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.best_loss = float('inf')

    def save_checkpoint(self, model: nn.Module, val_loss: float) -> None:
        if self.save_best_only:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                }, self.filepath)
        else:
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, self.filepath)

def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    config: TrainingConfig,
    device: torch.device
) -> Tuple[List[float], List[float], List[float]]:
    
    generator.train()
    discriminator.train()

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=config.learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config.learning_rate)
    
    scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', patience=5)
    scheduler_d = ReduceLROnPlateau(optimizer_d, mode='min', patience=5)
    
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    g_checkpoint = ModelCheckpoint('generator_best.pth')
    d_checkpoint = ModelCheckpoint('discriminator_best.pth')

    train_losses_g = []
    train_losses_d = []
    train_scores_d = []

    for epoch in range(config.epochs):
        epoch_loss_g = 0
        epoch_loss_d = 0
        epoch_score_d = 0

        generator.train()
        discriminator.train()

        for batch in dataloader:
            x_data, y_data, x_lengths, y_lengths = [b.to(device) for b in batch]

            # Train discriminator
            optimizer_d.zero_grad()
            real_labels = torch.ones(y_data.size(0), 1).to(device)
            fake_labels = torch.zeros(y_data.size(0), 1).to(device)

            real_loss = criterion(discriminator(y_data), real_labels)
            fake_data = generator(x_data)
            fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            
            d_loss.backward()
            clip_grad_norm_(discriminator.parameters(), config.clip_grad)
            optimizer_d.step()

            # Train generator
            optimizer_g.zero_grad()
            g_loss = criterion(discriminator(fake_data), real_labels)
            d_score = discriminator.score(fake_data).mean()
            
            total_loss = g_loss + d_score
            total_loss.backward()
            clip_grad_norm_(generator.parameters(), config.clip_grad)
            optimizer_g.step()

            epoch_loss_g += g_loss.item()
            epoch_loss_d += d_loss.item()
            epoch_score_d += d_score.item()

        # Validation phase
        generator.eval()
        discriminator.eval()
        val_loss_g = 0
        val_loss_d = 0

        with torch.no_grad():
            for batch in val_dataloader:
                x_data, y_data, x_lengths, y_lengths = [b.to(device) for b in batch]
                fake_data = generator(x_data)
                val_loss_g += criterion(discriminator(fake_data), real_labels).item()
                val_loss_d += criterion(discriminator(y_data), real_labels).item()

        val_loss_g /= len(val_dataloader)
        val_loss_d /= len(val_dataloader)

        # Update schedulers
        scheduler_g.step(val_loss_g)
        scheduler_d.step(val_loss_d)

        # Save checkpoints
        g_checkpoint.save_checkpoint(generator, val_loss_g)
        d_checkpoint.save_checkpoint(discriminator, val_loss_d)

        # Early stopping
        if early_stopping(val_loss_g):
            logging.info("Early stopping triggered")
            break

        # Logging
        logging.info(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"D loss: {epoch_loss_d/len(dataloader):.4f} | "
            f"G loss: {epoch_loss_g/len(dataloader):.4f} | "
            f"D score: {epoch_score_d/len(dataloader):.4f} | "
            f"Val G loss: {val_loss_g:.4f} | "
            f"Val D loss: {val_loss_d:.4f}"
        )

        train_losses_g.append(epoch_loss_g / len(dataloader))
        train_losses_d.append(epoch_loss_d / len(dataloader))
        train_scores_d.append(epoch_score_d / len(dataloader))

    return train_losses_g, train_losses_d, train_scores_d

def visualize_training(losses_g: List[float], losses_d: List[float], scores_d: List[float]) -> None:
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses_g)
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(losses_d)
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(scores_d)
    plt.title('Discriminator Scores')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    
    plt.tight_layout()
    plt.savefig('training_visualization.png')
    plt.close()

def main():
    set_seed(42)
    
    # Initialize environment
    env = Environment(temperature=25.0, pH=7.0, chemical_exposure=0.5, radiation=0.0)
    env.validate()
    
    # Configuration
    config = TrainingConfig()
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        # Load dataset
        x_data, y_data, dna_purpose = load_dataset("train.csv", env)
        x_data_val, y_data_val = load_validation_dataset("val.csv", env)

        # Create DataLoaders
        train_dataset = CustomDataset(x_data, y_data)
        val_dataset = CustomDataset(x_data_val, y_data_val)
        
        dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        dataloader_val = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        # Initialize models
        tokenizer = DNATokenizer()
        input_dim = len(tokenizer.token2idx)
        generator = Generator(input_dim, config.hidden_dim, input_dim, env).to(device)
        discriminator = Discriminator(input_dim, config.hidden_dim, env).to(device)

        # Train models
        logging.info("Starting training...")
        train_losses_g, train_losses_d, train_scores_d = train_gan(
            generator,
            discriminator,
            dataloader,
            dataloader_val,
            config,
            device
        )

        # Visualize training results
        visualize_training(train_losses_g, train_losses_d, train_scores_d)

        # Generate sample sequences
        logging.info("Generating sample sequences...")
        generator.eval()
        discriminator.eval()
        
        num_samples = 10
        generated_seqs = []
        discriminator_scores = []

        with torch.no_grad():
            for _ in range(num_samples):
                noise = torch.randint(0, input_dim, (1, 100)).to(device)
                output = generator(noise)
                seq_indices = torch.argmax(output, dim=-1)[0]
                generated_seq = tokenizer.decode(seq_indices.cpu().tolist())
                generated_seqs.append(generated_seq)
                
                # Calculate discriminator score
                seq_tensor = torch.tensor([tokenizer.encode(generated_seq)]).to(device)
                score = discriminator.score(seq_tensor).item()
                discriminator_scores.append(score)

        # Generate report
        logging.info("\nGenerated DNA Sequences Report:")
        logging.info("================================")
        logging.info(f"\nDNA Purpose: {dna_purpose}")
        logging.info("\nGenerated Sequences:")
        
        for i, (seq, score) in enumerate(zip(generated_seqs, discriminator_scores)):
            logging.info(f"\nSequence {i+1}: {seq}")
            logging.info(f"Discriminator Score: {score:.4f}")
            
            # Basic sequence analysis
            gc_content = (seq.count('G') + seq.count('C')) / len(seq) * 100
            logging.info(f"GC Content: {gc_content:.2f}%")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
