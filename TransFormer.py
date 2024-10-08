from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import pandas as pd

df = pd.read_csv("combined_dialogue_data.csv")
texts = df["dialogue"].tolist()
labels = df["label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        item['labels'] = self.labels[idx]
        return item

train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)

model = BertForSequenceClassification.from_pretrained('distilbert-base-uncased')

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

trainer.save_model("./ai_human_detector")
