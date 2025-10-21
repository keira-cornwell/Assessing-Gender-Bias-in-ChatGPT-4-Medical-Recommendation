import os
import json
import torch
import numpy as np
import random
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from collections import Counter

def label_responses():
    with open("final_gender_bias_diagnosis_results.json", "r") as one_datafile:
        one_data = json.load(one_datafile)
    data_vector = []
    for entry in one_data:
        response_dict = {}
        response_dict["gender"]= entry["gender"]
        response_dict["response"]= entry["response"]
        data_vector.append(response_dict)
    
    with open("two_final_gender_bias_diagnosis_results.json", "r") as two_datafile:
        two_data = json.load(two_datafile)
    for entry in two_data:
        response_dict = {}
        response_dict["gender"]= entry["gender"]
        response_dict["response"]= entry["response"]
        data_vector.append(response_dict)
    
    responses = []
    labels = []
    for entry in data_vector:
        responses.append(
            entry["response"]
            .replace("females", "people")
            .replace("males", "people")
            .replace("women", "people")
            .replace("men", "people")
        )
        if entry["gender"] == "female":
            labels.append(1)
        else:
            labels.append(0)
    return responses, labels 

class TextClassificationDataset(Dataset):
    def __init__ (self, responses, labels, tokenizer, max_length):
        self.responses = responses
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.responses)
    
    def __getitem__(self, idx):
        response = self.responses[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(response, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}
    
class BERTclassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTclassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)  
        return logits

def train_model(model, data_loader, optimizer, scheduler, device, loss_fn):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)  # ✅ Add this
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

import torch

def predict_gender(text, model, tokenizer, device, max_length=256):
    model.eval()
    
    encoding = tokenizer(
        text,
        return_tensors='pt',
        max_length=max_length,
        padding='max_length',
        truncation=True
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

    return "woman" if preds.item() == 1 else "man"

responses, labels = label_responses()
bert_model_name = "bert-base-uncased"
num_classes = 2
max_length = 256
batch_size = 16
num_epochs = 4
learning_rate = 0.00002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_responses, val_responses, train_labels, val_labels = train_test_split(responses, labels, test_size=0.2, random_state=42)

class_counts = Counter(train_labels)
# Inverse of class frequency (more weight for underrepresented/misclassified class)
weights = torch.tensor(
    [1.0 / class_counts[0], 1.0 / class_counts[1]],
    dtype=torch.float
).to(device)

loss_fn = nn.CrossEntropyLoss(weight=weights)

tokenizer = BertTokenizer.from_pretrained(bert_model_name)
train_dataset = TextClassificationDataset(train_responses, train_labels, tokenizer, max_length)
val_dataset = TextClassificationDataset(val_responses, val_labels, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = BERTclassifier(bert_model_name, num_classes).to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_model(model, train_loader, optimizer, scheduler, device, loss_fn)
    val_accuracy, val_report = evaluate(model, val_loader, device)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print("Classification Report:\n", val_report)

# Save the model
torch.save(model.state_dict(), "bert_classifier.pth")
