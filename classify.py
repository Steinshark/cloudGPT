import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import json
from model import LMSteinshark
from data import load_tokenizer
import random 
# === CONFIG ===
INPUT_FILE = r"//Steinpc/s/nlp/data/factual_dataset.jsonl"
MODEL_PATH = "//Steinpc/s/nlp/models/PreFinetune352"
TOKENIZER_PATH = "//Steinpc/s/nlp/tokenizer"
DATA_FILE = "//Steinpc/s/nlp/data/relevant_topics.jsonl"
CLASSIFIER_HEAD_PATH = "classifier_head.pth"

BATCH_SIZE = 16
LR = 2e-4
EPOCHS = 4
DEVICE = "cuda"


def get_class_weights(input_file):
    """
    Compute class weights based on the dataset.
    More negative labels -> higher weight for positive class.
    Returns a tensor to pass to CrossEntropyLoss.
    """
    counts = [0, 0]  # [negatives, positives]
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                label = entry["label"]
                counts[label] += 1
            except Exception:
                continue

    total = sum(counts)
    weights = [total / (2 * c) if c > 0 else 1.0 for c in counts]  # inverse frequency
    return torch.tensor(weights, dtype=torch.bfloat16)

# Example usage in training:
weights = get_class_weights(DATA_FILE).to(DEVICE)
criterion = nn.CrossEntropyLoss()#weight=weights)
#criterion = nn.BCEWithLogitsLoss()#pos_weight=weights[0]/weights[1])


# === ADD CLASSIFICATION HEAD ===
class Classifier(nn.Module):
    def __init__(self, lm_model:LMSteinshark, hidden_size=2048):  # adjust hidden_size to your model
        super().__init__()
        self.lm = lm_model
        self.inter  = nn.Linear(hidden_size,hidden_size*2)
        self.head = nn.Linear(hidden_size*2, 2)  # 2 classes: relevant / not relevant



    def forward(self, tokens, mask=None):
        # tokens: [B, L]
        output  = self.lm.forward_no_head(tokens, tokens, mask)  # assuming your model outputs [B,L,H]
        hidden  = output[:, -1, :]  # [B,H]
        hidden  = self.inter(hidden)
        hidden  = torch.nn.functional.relu(hidden) 
        # take last token hidden state (or mean)
        logits = self.head(hidden)  # [B,2]
        return logits


def classify(classifier:Classifier,tokenizer,article:str):
    tokens          = torch.tensor(tokenizer.encode(article).ids).unsqueeze(0).long().to(DEVICE)
    mask            = torch.ones_like(tokens, dtype=torch.bool)

    with torch.inference_mode():
        logits      = classifier(tokens, mask)
        probs       = torch.softmax(logits, dim=-1)
        pred_label  = torch.argmax(probs, dim=-1).item()
        confidence  = probs[0, pred_label].item() * 100 
    
    return pred_label, confidence

def load_model():
    lm_model = LMSteinshark.from_loadpoint(MODEL_PATH, p_override=0)
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    for p in lm_model.parameters():
        p.requires_grad_(False)
    lm_model = lm_model.eval().to(DEVICE, dtype=torch.bfloat16)

    classifier = Classifier(lm_model).to(DEVICE, dtype=torch.bfloat16)
    classifier.head.load_state_dict(torch.load(CLASSIFIER_HEAD_PATH, map_location=DEVICE))
    classifier.eval()
    return classifier, tokenizer

def sample_predictions(n=5):
    classifier, tokenizer = load_model()
    # Count lines in file
    with open(INPUT_FILE, "r", encoding="utf-8") as f:

        for _ in range(n):
            # Pick a random line
            for _ in range(random.randint(50,400)):
                f.readline() 
            data = json.loads(f.readline())

            pred_label, confidence  = classify(classifier,tokenizer,data)

            print("\n--- Sample ---")
            print("Text (up to <response>):", text)
            print(f"Predicted label: {pred_label} ({confidence*100:.2f}% confidence)")

# === DATASET ===
class WikiDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.data[idx]["label"]
        tokens = torch.tensor(tokenizer.encode(text).ids).long()
        return tokens, torch.tensor(label).long()

def collate_fn(batch):
    tokens_batch, labels_batch = zip(*batch)
    lengths = [t.size(0) for t in tokens_batch]
    max_len = max(lengths)
    padded = []
    mask = []
    for t in tokens_batch:
        pad_len = max_len - t.size(0)
        padded.append(torch.cat([t, torch.zeros(pad_len, dtype=torch.long)]))
        mask.append(torch.cat([torch.ones(t.size(0), dtype=torch.bool),
                               torch.zeros(pad_len, dtype=torch.bool)]))
    return torch.stack(padded).to(DEVICE), torch.stack(mask).to(DEVICE), torch.stack(labels_batch).to(DEVICE)

if __name__ == "__main__":
    # sample_predictions(n=20)
    # exit()

    # === LOAD MODEL & TOKENIZER ===
    model = LMSteinshark.from_loadpoint(MODEL_PATH, p_override=0)
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    # Freeze LM weights
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.eval().to(DEVICE, dtype=torch.bfloat16)
    classifier = Classifier(model).to(DEVICE, dtype=torch.bfloat16)

    dataset = WikiDataset(DATA_FILE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # === OPTIMIZER & LOSS ===
    optimizer = torch.optim.AdamW(list(classifier.head.parameters())+list(classifier.inter.parameters()), lr=LR)
    #criterion = nn.CrossEntropyLoss()

    # === TRAIN LOOP ===
    for epoch in range(EPOCHS):
        total_loss = 0
        for tokens, mask, labels in loader:
            optimizer.zero_grad()
            logits = classifier(tokens, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}")

    # === SAVE CLASSIFIER HEAD ===
    torch.save(classifier.head.state_dict(), "classifier_head.pth")
    print("✅ Classifier head saved.")
