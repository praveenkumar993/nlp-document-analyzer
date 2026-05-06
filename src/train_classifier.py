# ---- Custom Dataset ----
class AGNewsDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        text = example["text"]
        label = example["label"]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
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
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(
        all_labels, all_preds,
        target_names=LABEL_NAMES,
        zero_division=0
    )
    return accuracy, f1, report
# ---- Main ----
print("Loading dataset...")
dataset = load_dataset("ag_news")

# Use subset for faster training - still 30k samples
train_data = dataset["train"].select(range(30000))
test_data = dataset["test"]

print("Loading tokenizer and model...")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)
model.to(DEVICE)

print("Creating datasets...")
train_dataset = AGNewsDataset(train_data, tokenizer, MAX_LENGTH)
test_dataset = AGNewsDataset(test_data, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

best_f1 = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 40)

    train_loss = train(model, train_loader, optimizer, DEVICE, GRADIENT_ACCUMULATION_STEPS)
    print(f"Train Loss: {train_loss:.4f}")

    print("Evaluating...")
    accuracy, f1, report = evaluate(model, test_loader, DEVICE)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1: {f1:.4f}")
    print(report)

    if f1 > best_f1:
        best_f1 = f1
        os.makedirs("classifier_model", exist_ok=True)
        model.save_pretrained("classifier_model")
        tokenizer.save_pretrained("classifier_model")
        print(f"Model saved with F1: {best_f1:.4f}")

print(f"\nBest Test F1: {best_f1:.4f}")
print("Training complete!")
