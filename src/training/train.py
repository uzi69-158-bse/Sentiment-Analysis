import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.sentiment_model import SentimentClassifier
from src.data.data_ingestion import DataIngestion
from src.evaluation.metrics import compute_metrics
from utils.helpers import load_config
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_config = config['model']  # access model configuration
        self.model = SentimentClassifier(n_classes=model_config['num_classes']).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        learning_rate = float(config['training']['learning_rate'])
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config['training']['scheduler_step_size'], gamma=config['training']['scheduler_gamma'])

    def prepare_dataloader(self, dataset, batch_size, shuffle=True):
        # Ensure the labels are in integer format
        label_map = {
            'positive': 0,
            'negative': 1,
            'neutral': 2
        }

        # Map string labels to integers
        dataset['labels'] = [label_map[label] for label in dataset['labels']]  # Convert labels to integers

        # Ensure that input_ids, attention_mask, and labels are already lists
        input_ids = dataset['input_ids']  # Assuming it's already a list
        attention_mask = dataset['attention_mask']  # Assuming it's already a list
        labels = dataset['labels']  # Convert to list if not already

        # Print the first few elements to check if the conversion worked
        print(f"Sample input_ids: {input_ids[:5]}")
        print(f"Sample attention_mask: {attention_mask[:5]}")
        print(f"Sample labels: {labels[:5]}")

        # Ensure the lengths of input_ids, attention_mask, and labels match
        assert len(input_ids) == len(attention_mask) == len(labels), "Mismatch in data lengths"

        # Convert to PyTorch tensors
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)  # Convert labels to integer tensor

        # Create a TensorDataset
        data = TensorDataset(input_ids_tensor, attention_mask_tensor, labels_tensor)

        # Create DataLoader
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)

        return dataloader



    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        metrics = compute_metrics(all_labels, all_preds)
        return avg_loss, metrics
    
    def train(self, train_loader, val_loader, num_epochs):
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            print("Training Started...")
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Metrics: {val_metrics}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config['training']['model_save_path'])
                print("Saved best model")
        
        print("Training completed")

if __name__ == "__main__":
    config = load_config('config/config.yaml')
    
    data_ingestion = DataIngestion(config['data'])
    train_data, val_data, _ = data_ingestion.process_data()
    
    trainer = Trainer(config)
    train_loader = trainer.prepare_dataloader(train_data, config["training"]['batch_size'])
    val_loader = trainer.prepare_dataloader(val_data, config["training"]['batch_size'])
    
    trainer.train(train_loader, val_loader, config['training']['num_epochs'])
