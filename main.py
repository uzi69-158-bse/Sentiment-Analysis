import torch
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.sentiment_model import SentimentClassifier
from src.data.data_ingestion import DataIngestion
from src.training.train import Trainer
from utils.helpers import load_config
def analyze_dataset(trainer, data):
    """Analyze sentiments across the dataset and visualize results."""
    print("Analyzing dataset...")
    dataloader = trainer.prepare_dataloader(data, batch_size=32, shuffle=False)
    all_preds = []
    all_labels = data['labels']

    trainer.model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, _ = [b.to(trainer.device) for b in batch]
            outputs = trainer.model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())

    sentiment_counts = pd.Series(all_preds).value_counts(sort=False)
    sentiment_counts.index = ['Positive', 'Negative', 'Neutral'][:len(sentiment_counts)]
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'], title="Sentiment Analysis")
    plt.ylabel("Count")
    plt.show()

def single_text_sentiment(model, tokenizer, text):
    """Analyze sentiment of a single input text."""
    tokens = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt')
    input_ids = tokens['input_ids'].to(model.device)
    attention_mask = tokens['attention_mask'].to(model.device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1).cpu().item()

    sentiments = {0: "Positive", 1: "Negative", 2: "Neutral"}
    print(f"Sentiment for the given text: {sentiments[preds]}")

def sentiment_ratio_analysis(data):
    """Visualize sentiment distribution ratio."""
    sentiment_counts = pd.Series(data['labels']).value_counts(normalize=True) * 100
    sentiment_counts.index = ['Positive', 'Negative', 'Neutral'][:len(sentiment_counts)]
    sentiment_counts.plot.pie(autopct="%1.1f%%", colors=['green', 'red', 'blue'], title="Sentiment Distribution")
    plt.ylabel("")  # Hide y-label for better visualization
    plt.show()

def main():
    # Load configuration
    print("Loading configuration...")
    config = load_config('config/config.yaml')

    # Data ingestion and processing
    print("Processing data...")
    data_ingestion = DataIngestion(config['data'])
    train_data, val_data, test_data = data_ingestion.process_data()

    # Initialize the SentimentClassifier (using the pre-initialized model and tokenizer)
    print("Initializing sentiment model...")
    model = SentimentClassifier(config['model']['num_classes'])  # Or switch to LSTMSentimentClassifier
    tokenizer = model.tokenizer  # Access the tokenizer if already initialized within the class

    # Trainer initialization
    print("Initializing trainer...")
    trainer = Trainer(config)

    while True:
        # Display menu options for the user
        print("\nChoose an option:")
        print("1. Analyze the entire dataset.")
        print("2. Analyze the sentiment of a single text input.")
        print("3. Visualize sentiment ratio in the dataset.")
        print("4. Exit.")

        try:
            choice = int(input("Enter your choice: "))
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 4.")
            continue

        # Handle user's choice
        if choice == 1:
            print("Dataset analysis selected.")
            analyze_dataset(trainer, test_data)

        elif choice == 2:
            text = input("Enter the text for sentiment analysis: ")
            single_text_sentiment(model, tokenizer, text)

        elif choice == 3:
            print("Visualizing sentiment ratio...")
            sentiment_ratio_analysis(test_data)

        elif choice == 4:
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
