
from tqdm import tqdm

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset  # DL = in batches, TD = create dataset from tensors

# from transformers import AutoTokenizer, AutoModelForSequenceClassification  # Not needed
from transformers import BertTokenizer, BertForSequenceClassification  # Tokenizing

from torch.optim import AdamW  # Optimizer

# load dataset
dataset = load_dataset("imdb")

# Extract training/testing text & labels
train_texts = dataset['train']['text']
train_labels = dataset['train']['label']

test_texts = dataset['test']['text']
test_labels = dataset['test']['label']

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # pre-trained
# Pre-trained model / two possible labels
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize and encode the training and testing texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

# Create PyTorch dataset
train_dataset = TensorDataset(
    train_encodings['input_ids'],
    train_encodings['attention_mask'],
    torch.tensor(train_labels)
)

# Create PyTorch datasets
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'],
                              torch.tensor(train_labels))
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_labels))

# DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# optimizer and device
optimizer = AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to device
model.to(device)

# Fine-tuning loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    train_correct = 0
    train_total = 0

    # tqdm to show progress
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)
        train_total += labels.size(0)
        train_correct += (predicted_labels == labels).sum().item()

    train_accuracy = train_correct / train_total
    print(f'Epoch {epoch + 1}/{num_epochs}, Training Accuracy: {train_accuracy}')



    # Evaluation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1)

            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()

        accuracy = correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {accuracy}')

        # Test 1 - Epoch 1/1, Test Accuracy: 0.5 (50%)

        # -----------------------------------------------

        # Test 2 Updates:
        # Increased Epochs from 1 -> 3, and added training accuracy printing.

        # Test 2 Results:
        # Epoch 1/3, Training Accuracy: 0.50396 (50.396%), Test Accuracy: 0.5 (50%)
        # Epoch 2/3, Training Accuracy: 0.5058 (50.58%), Test Accuracy: 0.5 (50%)
        # Epoch 3/3, Training Accuracy: 0.50244 (50.244%), Test Accuracy: 0.5 (50%)

        # -----------------------------------------------

        # Test 3 Updates:
        # Changed optimizer from SGD -> AdamW, learning rate (lr) changed from 0.01 -> 2e-5,
        # Number of epochs decreased from 3 -> 1, batch_size increased from 8 -> 32.

        # Test 3 Results:
        # Epoch 1/1, Training Accuracy: 0.90144 (90.144%), Test Accuracy: 0.92904 (92.904%)

        # -----------------------------------------------

        # Test 4 Updates:
        # Increased epochs from 1 -> 3

        # Test 4 Results:
        # Epoch 1/3, Training Accuracy: 0.904 (90.4%), Test Accuracy: 0.93664 (93.664)
        # Epoch 2/3, Training Accuracy: 0.95804 (95.804%), Test Accuracy: 0.93516 (93.516%)
        # Epoch 3/3, Training Accuracy: 0.97932 (97.932%), Test Accuracy: 0.94032 (94.032%)

        # -----------------------------------------------

        # Test 5 Updates:
        # Batch size increased from 32 -> 64:

        # Test 5 Results:
        # Epoch 1/3, Training Accuracy: 0.89676 (89.676%), Test Accuracy: 0.93468 (93.468%)
        # Epoch 2/3, Training Accuracy: 0.94932 (94.932%), Test Accuracy: 0.92684 (92.684%)
        # Epoch 3/3, Training Accuracy: 0.9758 (97.58%), Test Accuracy: 0.92192 (92.192%)


