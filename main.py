import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import os
import logging
from datetime import datetime
import argparse

# Configure logging: stdout + file with timestamps and levels
# Notes: Use logging instead of print for production-ready scripts.
#       RotatingFileHandler could be added for large training runs.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # stdout
        logging.FileHandler(f'char_rnn_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Check CUDA availability early
if not torch.cuda.is_available():
    logger.warning("CUDA not available. Using CPU. Install CUDA-enabled PyTorch for GPU acceleration.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


# === Model (optimized with dropout for regularization) ===
# Notes: Added dropout to prevent overfitting. Increased default hidden_dim.
#        Use batch_first=True consistently.
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x = self.dropout(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])  # Last timestep output
        return out, hidden


# === Data preparation (batched for efficiency) ===
# Notes: Added batching to leverage parallelism. Use DataLoader for shuffling.
#        Validate file existence early.
def load_text_and_prepare_dataset(file_path="training_data_german.txt", seq_length=50, batch_size=64):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training file '{file_path}' not found. Please create it with German text data.")

    logger.info(f"Loading data from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower()

    chars = sorted(list(set(text)))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    logger.info(f"Vocabulary size: {len(chars)} characters")

    data = [char2idx[c] for c in text]
    inputs, targets = [], []
    for i in range(len(data) - seq_length):
        inputs.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger.info(f"Dataset size: {len(dataset)} sequences")
    return dataloader, char2idx, idx2char


# === Model/Vocab persistence ===
def save_model(model, path="char_rnn_model.pt"):
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")


def load_model(model, path="char_rnn_model.pt", map_location=None):
    model.load_state_dict(torch.load(path, map_location=map_location))
    logger.info(f"Model loaded from {path}")


def save_vocab(char2idx, path="vocab.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(char2idx, f)
    logger.info(f"Vocab saved to {path}")


def load_vocab(path="vocab.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# === Training loop (batched, with validation-like loss tracking) ===
# Notes: Process in batches for speed. Added gradient clipping.
#        Early stopping or validation could be added next.
def train(model, dataloader, epochs=10, lr=0.001):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # AdamW + L2 reg
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Starting training for {epochs} epochs (lr={lr})")
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch_inputs, batch_targets in dataloader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            optimizer.zero_grad()
            outputs, _ = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.4f}")


# === Generation (top-k sampling for better diversity) ===
# Notes: Top-k sampling instead of pure multinomial for coherent output.
#        Fixed input_tensor shape handling.
def sample(model, char2idx, idx2char, start_str="hallo", length=100, top_k=50):
    model.eval()
    input_seq = [char2idx.get(ch, 0) for ch in start_str.lower()]
    input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)

    hidden = None
    generated = start_str

    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_tensor[:, -1:], hidden)  # Use last char only
            probs = F.softmax(output, dim=-1)
            # Top-k sampling
            probs_topk, topk_indices = torch.topk(probs, top_k)
            probs_topk = probs_topk / probs_topk.sum(dim=-1, keepdim=True)
            char_idx = torch.multinomial(probs_topk, 1).squeeze().item()
            char_idx_full = topk_indices[0, char_idx].item()
            char = idx2char.get(char_idx_full, '?')
            generated += char
            input_tensor = torch.tensor([[char_idx_full]], dtype=torch.long).to(device)

    return generated


# === CLI Interface ===
# Notes: Added argparse for flexibility (epochs, seq_length, etc.).
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Char-RNN German text generator")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--seq-length", type=int, default=50, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--data-file", default="training_data_german.txt", help="Training data file")
    args = parser.parse_args()

    # Load data
    dataloader, char2idx, idx2char = load_text_and_prepare_dataset(
        args.data_file, args.seq_length, args.batch_size
    )

    # Initialize model
    model = CharRNN(
        vocab_size=len(char2idx),
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.2
    ).to(device)

    # Load existing model/vocab if available
    model_exists = os.path.exists("char_rnn_model.pt")
    vocab_exists = os.path.exists("vocab.json")
    if model_exists and vocab_exists:
        logger.info("Loading existing model and vocabulary...")
        load_model(model, map_location=device)
        saved_vocab = load_vocab()
        # Rebuild idx2char if vocab mismatch
        if set(saved_vocab.keys()) == set(char2idx.keys()):
            char2idx = saved_vocab
            idx2char = {int(k): v for k, v in char2idx.items()}
        else:
            logger.warning("Vocab mismatch. Using new vocab.")
    else:
        logger.info("Creating new model. Training from scratch.")
        save_vocab(char2idx)

    # Training
    train(model, dataloader, epochs=args.epochs)

    # Save
    save_model(model)
    save_vocab(char2idx)

    logger.info("✅ Training completed. Model saved.")

    # Interactive generation
    logger.info("🤖 Interactive mode. Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            break
        if user_input:
            output = sample(model, char2idx, idx2char, start_str=user_input, length=100)
            print(f"AI: {output}")
