import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from nltk_utils import tokenize, stem
from model import NeuralNet
from utils import bag_of_words

# Load intents from JSON
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Loop through each sentence in intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        words = self.x_data[index]
        label = self.y_data[index]
        words = [str(w) for w in words]  # Convert to strings
        words = ' '.join(words)  # Join the words into a string
        encoding = tokenizer.encode_plus(
            words,
            add_special_tokens=True,
            max_length=512,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return input_ids, attention_mask, label



    # def __getitem__(self, index):
    #     words = self.x_data[index]
    #     label = self.y_data[index]
    #     encoding = tokenizer.encode_plus(
    #         words,
    #         add_special_tokens=True,
    #         max_length=512,
    #         return_tensors='pt',
    #         padding='max_length',
    #         truncation=True
    #     )
    #     input_ids = encoding['input_ids'].squeeze()
    #     attention_mask = encoding['attention_mask'].squeeze()

    #     return input_ids, attention_mask, label

    def __len__(self):
        return self.n_samples

batch_size = 8  # Specify the desired batch size

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hidden_size = 8
output_size = len(tags)
model = NeuralNet(hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.001  # Specify the desired learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 1000
for epoch in range(num_epochs):
    for (input_ids, attention_mask, labels) in train_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        

        
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "hidden_size": hidden_size,
    "output_size": output_size,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete. Model saved to {FILE}')
