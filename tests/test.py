import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.sentiment_model import SentimentClassifier
from src.data.data_ingestion import DataIngestion
from src.evaluation.metrics import compute_metrics
from utils.helpers import load_config

class Tester:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_config = config['model']
        self.model = SentimentClassifier(n_classes=model_config['num_classes']).to(self.device)
        self.model.load_state_dict(torch.load(config['training']['model_save_path']))
        self.model.eval()


    def prepare_dataloader(self, dataset, batch_size):
        # Ensure dataset contains the required keys
        if not all(key in dataset for key in ['input_ids', 'attention_mask', 'labels']):
            raise ValueError("Dataset missing required keys: 'input_ids', 'attention_mask', 'labels'")
        
        input_ids = dataset['input_ids']
        attention_mask = dataset['attention_mask']
        labels = dataset['labels']
        
            # Mapping string labels to integers
        label_map = {"positive": 0, "negative": 1, "neutral": 2}
        labels = [label_map[label] for label in labels]  # Convert string labels to integers


        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        data = TensorDataset(input_ids_tensor, attention_mask_tensor, labels_tensor)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

        return dataloader

    def evaluate(self, test_loader):
        total_loss = 0
        all_preds = []
        all_labels = []
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(test_loader)
        metrics = compute_metrics(all_labels, all_preds)
        return avg_loss, metrics

if __name__ == "__main__":
    config = load_config('config/config.yaml')

    data_ingestion = DataIngestion(config['data'])
    _, _, test_data = data_ingestion.process_data()

    tester = Tester(config)
    test_loader = tester.prepare_dataloader(test_data, config['testing']['batch_size'])

    print("Starting evaluation...")
    test_loss, test_metrics = tester.evaluate(test_loader)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Metrics: {test_metrics}")
