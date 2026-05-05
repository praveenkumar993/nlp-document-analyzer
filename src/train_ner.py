import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from datasets import load_dataset
import numpy as np
import mlflow
import mlflow.pytorch
from sklearn.metrics import f1_score, classification_report
import os
from dotenv import load_dotenv

load_dotenv()

# ---- Config ----
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
GRADIENT_ACCUMULATION_STEPS = 2

LABEL_LIST = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---- Custom Dataset ----
class NERDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]

        # Tokenize with word ids to align labels
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Align labels with tokenized output
        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)  # special tokens ignored in loss
            elif word_id != previous_word_id:
                aligned_labels.append(ner_tags[word_id])  # first token of word
            else:
                aligned_labels.append(-100)  # subsequent tokens of same word ignored
            previous_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }
    

# ---- Training Function ----
def train(model, dataloader, optimizer, device, gradient_accumulation_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()
        total_loss += outputs.loss.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if step % 50 == 0:
            print(f"  Step {step}/{len(dataloader)} — Loss: {outputs.loss.item():.4f}")

    return total_loss / len(dataloader)


# ---- Evaluation Function ----
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=-1)

            # Only keep real tokens, ignore -100
            for pred_seq, label_seq in zip(preds, labels):
                for p, l in zip(pred_seq, label_seq):
                    if l.item() != -100:
                        all_preds.append(p.item())
                        all_labels.append(l.item())

    f1 = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(
        all_labels, all_preds,
        target_names=LABEL_LIST,
        zero_division=0
    )
    return f1, report
# ---- Main ----
if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_dataset("wikiann", "en")

    print("Loading tokenizer and model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    model.to(DEVICE)

    print("Creating datasets...")
    train_dataset = NERDataset(dataset["train"], tokenizer, MAX_LENGTH)
    val_dataset = NERDataset(dataset["validation"], tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # MLflow tracking
    mlflow.set_experiment("NER_DistilBERT")

    with mlflow.start_run():
        mlflow.log_param("model", MODEL_NAME)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("max_length", MAX_LENGTH)

        best_f1 = 0

        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            print("-" * 40)

            train_loss = train(
                model, train_loader, optimizer,
                DEVICE, GRADIENT_ACCUMULATION_STEPS
            )
            print(f"Train Loss: {train_loss:.4f}")

            print("Evaluating...")
            f1, report = evaluate(model, val_loader, DEVICE)
            print(f"Validation F1: {f1:.4f}")
            print(report)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_f1", f1, step=epoch)

            if f1 > best_f1:
                best_f1 = f1
                os.makedirs("models", exist_ok=True)
                model.save_pretrained("models/ner_model")
                tokenizer.save_pretrained("models/ner_model")
                print(f"Model saved with F1: {best_f1:.4f}")

        print(f"\nBest Validation F1: {best_f1:.4f}")
        mlflow.log_metric("best_f1", best_f1)

    print("Training complete!")