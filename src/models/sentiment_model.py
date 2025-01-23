import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, bert_model='bert-base-uncased', dropout=0.3, device=None):
        super(SentimentClassifier, self).__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device
        self.bert = BertModel.from_pretrained(bert_model)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)  # Initialize tokenizer here
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.attention = nn.MultiheadAttention(self.bert.config.hidden_size, num_heads=8)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Apply attention mechanism
        attn_output, _ = self.attention(pooled_output.unsqueeze(0), pooled_output.unsqueeze(0), pooled_output.unsqueeze(0))
        attn_output = attn_output.squeeze(0)
        
        x = self.dropout(attn_output)
        logits = self.fc(x)
        return logits

class LSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_classes, n_layers=2, dropout=0.3):
        super(LSTMSentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        return logits

def load_pretrained_embeddings(embedding_path):
    """
    Load pretrained word embeddings (GloVe or Word2Vec)
    """
    embeddings_index = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.FloatTensor(list(map(float, values[1:])))
            embeddings_index[word] = vector
    return embeddings_index

def create_embedding_matrix(word_index, embeddings_index, embedding_dim):
    """
    Create an embedding matrix for the vocabulary
    """
    embedding_matrix = torch.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

if __name__ == "__main__":
    bert_classifier = SentimentClassifier(n_classes=3)
    print(bert_classifier)
    
    vocab_size = 10000
    embedding_dim = 100
    hidden_dim = 256
    n_classes = 3
    
    lstm_classifier = LSTMSentimentClassifier(vocab_size, embedding_dim, hidden_dim, n_classes)
    print(lstm_classifier) 
