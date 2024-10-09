from transformers import DistilBertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import pandas as pd

# Load the dataset
df = pd.read_csv("combined_dialogue_data.csv")

# Extract the 'dialogue' and 'label' columns from the dataset
texts = df["dialogue"].tolist()
labels = df["label"].tolist()

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Save the training and testing sets into separate CSV files for reference
train_df = pd.DataFrame({"dialogue": train_texts, "label": train_labels})
test_df = pd.DataFrame({"dialogue": test_texts, "label": test_labels})

train_df.to_csv("train_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)

# Initialize the tokenizer for DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Dataset class for text data
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        # Tokenize the input texts, apply padding, and truncation
        self.inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        # Convert the labels to a tensor
        self.labels = torch.tensor(labels, dtype=torch.long)

    # Return the length of the dataset
    def __len__(self):
        return len(self.labels)

    # Retrieve a single item from the dataset by index
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        item['labels'] = self.labels[idx]
        return item

# Create instances of the custom dataset for training and testing
train_dataset = TextDataset(train_texts, train_labels)
test_dataset = TextDataset(test_texts, test_labels)

# Load a pre-trained DistilBERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Move the model to the GPU (if available) for faster computation
device = torch.device("cuda")
model.to(device)

# Define the training arguments such as the output directory, number of epochs, batch size, etc
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    learning_rate=2e-5
)

# Initialize the Trainer object with model, training arguments, datasets, etc
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the trained model to local
trainer.save_model("./ai_human_detector")
