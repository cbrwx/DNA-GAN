import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, temperature, pH, chemical_exposure, radiation):
        self.temperature = temperature
        self.pH = pH
        self.chemical_exposure = chemical_exposure
        self.radiation = radiation

class DNATokenizer:
    def __init__(self, environment):
        self.token2idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '<EOS>': 4}
        self.idx2token = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: '<EOS>'}
        self.environment = environment

    def encode(self, dna_sequence):
        return [self.token2idx[token] for token in dna_sequence] + [self.token2idx['<EOS>']]

    def decode(self, encoded_sequence):
        return [self.idx2token[idx] for idx in encoded_sequence[:-1]]

def load_dataset(csv_file, environment):
    with open(csv_file, 'r') as f:
        dna_purpose = f.readline().strip()
    dataset = []
    with open(csv_file, 'r') as f:
        for line in f:
            if line.startswith(">"):
                continue
            line = line.strip()
            if len(line) > 0:
                dataset.append(line)
    x_data, y_data = tokenize_and_encode(dataset, environment)
    return x_data, y_data, dna_purpose

def load_validation_dataset(csv_file, environment):
    with open(csv_file, 'r') as f:
        dna_purpose = f.readline().strip()
    dataset = []
    with open(csv_file, 'r') as f:
        for line in f:
            if line.startswith(">"):
                continue
            line = line.strip()
            if len(line) > 0:
                dataset.append(line)
    x_data, y_data = tokenize_and_encode(dataset, environment)
    return x_data, y_data

def tokenize_and_encode(dataset, environment):
    tokenizer = DNATokenizer(environment)
    x_data = []
    y_data = []

    for dna_seq in dataset:
        tokens = tokenizer.encode(dna_seq)
        x_data.append(tokens[:-1])
        y_data.append(tokens[1:])

    return x_data, y_data

class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        x_length = len(x)
        y_length = len(y)
        return x, y, x_length, y_length

def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    x_data, y_data, x_lengths, y_lengths = zip(*batch)
    x_data_padded = pad_sequence(x_data, padding_value=0, batch_first=True)
    y_data_padded = pad_sequence(y_data, padding_value=0, batch_first=True)
    return x_data_padded, y_data_padded, torch.tensor(x_lengths), torch.tensor(y_lengths)

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, environment):
        super(Generator, self).__init__()
        self.environment = environment
        self.env_dim = 4  # Number of environmental factors
        self.fc1 = nn.Linear(self.env_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        env_input = torch.tensor([self.environment.temperature, self.environment.pH, self.environment.chemical_exposure, self.environment.radiation], dtype=torch.float).unsqueeze(0).to(x.device)
        env_out = self.relu(self.fc1(env_input))
        x = torch.cat((x, env_out), dim=1)
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, environment):
        super(Discriminator, self).__init__()
        self.environment = environment
        self.env_dim = 4  # Number of environmental factors
        self.fc1 = nn.Linear(input_dim + self.env_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        env_input = torch.tensor([self.environment.temperature, self.environment.pH, self.environment.chemical_exposure, self.environment.radiation], dtype=torch.float).unsqueeze(0).repeat(x.size(0), 1).to(x.device)
        x = torch.cat((x, env_input), dim=1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

    def score(self, x):
        # Evaluate the discriminator score for the generated sequences
        return self.discriminator(x)

def train_gan(generator, discriminator, dataloader, epochs, device, clip_grad=1.0, lr_scheduler=None, early_stopping=None):
    generator.train()
    discriminator.train()

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters())
    optimizer_d = optim.Adam(discriminator.parameters())

    train_losses_g = []
    train_losses_d = []
    train_scores_d = []

    best_loss = float('inf')
    stop_counter = 0

    for epoch in range(epochs):
        epoch_loss_g = 0
        epoch_loss_d = 0
        epoch_score_d = 0

        for i, data in enumerate(dataloader):
            x_data, y_data, x_lengths, y_lengths = data
            x_data = x_data.to(device)
            y_data = y_data.to(device)

            real_labels = torch.ones(y_data.size(0), 1).to(device)
            fake_labels = torch.zeros(y_data.size(0), 1).to(device)

            # Train the discriminator
            optimizer_d.zero_grad()
            real_loss = criterion(discriminator(y_data), real_labels)
            fake_data = generator(x_data)
            fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            clip_grad_norm_(discriminator.parameters(), clip_grad)
            optimizer_d.step()

            # Train the generator
            optimizer_g.zero_grad()
            g_loss = criterion(discriminator(fake_data), real_labels)
            d_score = discriminator.score(fake_data).mean()
            total_loss = g_loss + d_score
            total_loss.backward()
            clip_grad_norm_(generator.parameters(), clip_grad)
            optimizer_g.step()

            epoch_loss_g += g_loss.item()
            epoch_loss_d += d_loss.item()
            epoch_score_d += d_score.item()

        avg_epoch_loss_g = epoch_loss_g / len(dataloader)
        avg_epoch_loss_d = epoch_loss_d / len(dataloader)
        avg_epoch_score_d = epoch_score_d / len(dataloader)

        train_losses_g.append(avg_epoch_loss_g)
        train_losses_d.append(avg_epoch_loss_d)
        train_scores_d.append(avg_epoch_score_d)

        print(f"Epoch {epoch+1}/{epochs} | D loss: {avg_epoch_loss_d:.4f} | G loss: {avg_epoch_loss_g:.4f} | D score: {avg_epoch_score_d:.4f}")

        # Evaluate the model on the validation set
        if epoch % 10 == 0:
            validation_losses = []
            generator.eval()
            discriminator.eval()
            with torch.no_grad():
                for data in dataloader_val:
                    x_data, y_data, x_lengths, y_lengths = data
                    x_data = x_data.to(device)
                    y_data = y_data.to(device)

                    fake_data = generator(x_data)
                    loss = criterion(discriminator(fake_data), real_labels)
                    validation_losses.append(loss.item())

            avg_validation_loss = sum(validation_losses) / len(validation_losses)
            print(f"Validation loss: {avg_validation_loss:.4f}")

        # Update the best model based on validation loss
        if avg_validation_loss < best_val_loss:
            best_val_loss = avg_validation_loss
            best_generator_state_dict = generator.state_dict()
            best_discriminator_state_dict = discriminator.state_dict()

        # Update the learning rate scheduler
        if lr_scheduler is not None:
            lr_scheduler.step(avg_epoch_loss_g)

    # Restore the best model and save it
    generator.load_state_dict(best_generator_state_dict)
    discriminator.load_state_dict(best_discriminator_state_dict)
    save_model(generator, "generator.pth")
    save_model(discriminator, "discriminator.pth")

    # Visualize the generated DNA sequences
    num_samples = 10
    generated_seqs = []
    for i in range(num_samples):
        noise = torch.randn(1, input_dim).to(device)
        generated_seq = []
        with torch.no_grad():
            for j in range(100):
                output = generator(noise)
                _, topi = output.max(2)
                token = topi.item()
                if token == DNATokenizer().token2idx['<EOS>']:
                    break
                generated_seq.append(token)
                noise = output
        dna_seq = DNATokenizer().decode(generated_seq)
        generated_seqs.append(''.join(dna_seq))
        print(f"Generated DNA sequence {i+1}: {generated_seqs[-1]}")

    # Compute discriminator score for each generated sequence
    discriminator_scores = []
    for seq in generated_seqs:
        encoded_seq = torch.tensor([DNATokenizer().encode(seq)]).to(device)
        score = discriminator.score(encoded_seq).item()
        discriminator_scores.append(score)

    # Output report with generated sequences and discriminator scores
    print("\nGenerated DNA Sequences Report:")
    print("================================\n")
    print(f"DNA Purpose: {dna_purpose}\n")
    print("Generated Sequences:\n")
    for i, seq in enumerate(generated_seqs):
        print(f"Sequence {i+1}: {seq}")
        print(f"Discriminator Score: {discriminator_scores[i]}\n")
