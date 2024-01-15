# Import necessary libraries
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torch.optim.lr_scheduler import StepLR
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class BERTDataset:
    def __init__(self, texts, labels, max_len=256):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.num_examples = len(self.texts)
    
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
      text = str(self.texts[idx])
      # Preprocess the text
      preprocessed_text = preprocess_text(text)
      label = self.labels[idx]
      tokenized_text = self.tokenizer(
          preprocessed_text,
          add_special_tokens=True,
          padding="max_length",
          max_length=self.max_len,
          truncation=True,
          return_tensors='pt'
      )
      ids = tokenized_text["input_ids"].squeeze()
      mask = tokenized_text["attention_mask"].squeeze()
      token_type_ids = tokenized_text["token_type_ids"].squeeze()

      return {
        "ids": ids,
        "mask": mask,
        "token_type_ids": token_type_ids,
        "target": label.clone().detach().requires_grad_(False),   
    }


class BertCNNToxicModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=6, out_channels=100, kernel_sizes=[2, 3, 4]):
        super(BertCNNToxicModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, return_dict=False)
        self.kernel_sizes = kernel_sizes  # Save kernel_sizes as an instance attribute
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, out_channels, (k, self.bert.config.hidden_size)) for k in self.kernel_sizes]
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(len(self.kernel_sizes) * out_channels, num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask):
      _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
      # Reshape pooled_output to have a dummy height dimension
      x = pooled_output.unsqueeze(1).unsqueeze(2)  # Now x has shape (batch_size, 1, 1, hidden_size)
      # Add a dummy height dimension that is at least as large as the largest kernel size
      x = x.expand(-1, -1, max(self.kernel_sizes), -1)  # Now x has shape (batch_size, 1, max(kernel_sizes), hidden_size)
      x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
      x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

      x = torch.cat(x, 1)
      x = self.dropout(x)
      logits = self.classifier(x)
      return logits
    

class BertToxicModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=6):
        super(BertToxicModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, return_dict=False)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize text
    lemmatizer = WordNetLemmatizer()
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in tokens])
    return lemmatized_output

def calculate_metrics(logits, targets):
    preds = torch.sigmoid(logits).cpu().detach().numpy().round()
    targets = targets.cpu().detach().numpy()
    
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average='micro', zero_division=1)
    recall = recall_score(targets, preds, average='micro', zero_division=1)
    f1 = f1_score(targets, preds, average='micro', zero_division=1)
    
    return accuracy, precision, recall, f1

from torch.utils.tensorboard import SummaryWriter

def train_and_evaluate(model, train_data_loader, valid_data_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    writer = SummaryWriter()  

    train_losses = []
    valid_losses = []
    train_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    valid_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            optimizer.zero_grad()
            ids = batch["ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)
            logits = model(ids, token_type_ids, mask)
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * ids.size(0)

            # Calculate training metrics
            train_acc, train_prec, train_rec, train_f1 = calculate_metrics(logits, targets)
            train_metrics['accuracy'].append(train_acc)
            train_metrics['precision'].append(train_prec)
            train_metrics['recall'].append(train_rec)
            train_metrics['f1'].append(train_f1)

        train_loss /= len(train_data_loader.dataset)
        train_losses.append(train_loss)

        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', np.mean(train_metrics['accuracy']), epoch)
        writer.add_scalar('Precision/Train', np.mean(train_metrics['precision']), epoch)
        writer.add_scalar('Recall/Train', np.mean(train_metrics['recall']), epoch)
        writer.add_scalar('F1/Train', np.mean(train_metrics['f1']), epoch)

        scheduler.step()

        model.eval()
        valid_loss = 0
        for batch in tqdm(valid_data_loader, desc=f"Epoch {epoch+1}/{epochs} Validation"):
            ids = batch["ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)
            logits = model(ids, token_type_ids, mask)
            loss = criterion(logits, targets)

            valid_loss += loss.item() * ids.size(0)

            # Calculate validation metrics
            valid_acc, valid_prec, valid_rec, valid_f1 = calculate_metrics(logits, targets)
            valid_metrics['accuracy'].append(valid_acc)
            valid_metrics['precision'].append(valid_prec)
            valid_metrics['recall'].append(valid_rec)
            valid_metrics['f1'].append(valid_f1)

        valid_loss /= len(valid_data_loader.dataset)
        valid_losses.append(valid_loss)

        # Log validation metrics to TensorBoard
        writer.add_scalar('Loss/Valid', valid_loss, epoch)
        writer.add_scalar('Accuracy/Valid', np.mean(valid_metrics['accuracy']), epoch)
        writer.add_scalar('Precision/Valid', np.mean(valid_metrics['precision']), epoch)
        writer.add_scalar('Recall/Valid', np.mean(valid_metrics['recall']), epoch)
        writer.add_scalar('F1/Valid', np.mean(valid_metrics['f1']), epoch)

    writer.close()

    return train_losses, valid_losses, train_metrics, valid_metrics


if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('data/train.csv').head(1000)
    epochs = 11
    # Select all target columns for the labels
    target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    labels = df[target_columns].values

    # Split the data into training and validation sets (80% train, 20% validation)
    df_train, df_valid, train_labels, valid_labels = train_test_split(
        df.comment_text.values, 
        labels, 
        test_size=0.20,  # 20% for validation
        random_state=42  # Seed for reproducibility
    )

    # Create datasets
    train_dataset = BERTDataset(df_train, train_labels)
    valid_dataset = BERTDataset(df_valid, valid_labels)

    # Create data loaders
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Initialize the model
    model = BertToxicModel()

    # Train the model and evaluate
    train_losses, valid_losses, train_metrics, valid_metrics = train_and_evaluate(model, train_data_loader, valid_data_loader, epochs=epochs)

    # Save the model
    torch.save(model.state_dict(), 'bert_toxic_model.pth')
    print("Model saved successfully")