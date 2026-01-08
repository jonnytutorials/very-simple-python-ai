import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

if not torch.cuda.is_available():
    print("You are currently using your CPU for the training of this Model. Consider installing CUDA, if you have a NVIDIA GPU, that supports it!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# === Modell ===
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])  # Ausgabe des letzten Zeitschritts
        return out, hidden

# === Text vorbereiten ===
def load_text_and_prepare_tensor(file_path="training_data_german.txt", seq_length=50):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower()

    # Alle Zeichen, auch Umlaute, Sonderzeichen, Zahlen, Leerzeichen
    chars = sorted(list(set(text)))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for ch, i in char2idx.items()}

    data = [char2idx[c] for c in text if c in char2idx]
    input_sequences = []
    target_sequences = []

    for i in range(len(data) - seq_length):
        input_seq = data[i:i+seq_length]
        target = data[i+seq_length]
        input_sequences.append(input_seq)
        target_sequences.append(target)

    X = torch.tensor(input_sequences, dtype=torch.long)  # long für Embedding
    Y = torch.tensor(target_sequences, dtype=torch.long)

    return X, Y, char2idx, idx2char

# === Modell speichern / laden ===
def save_model(model, path="char_rnn_model.pt"):
    torch.save(model.state_dict(), path)

def load_model(model, path="char_rnn_model.pt"):
    model.load_state_dict(torch.load(path, map_location=device))

def save_vocab(vocab, path="vocab.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f)

def load_vocab(path="vocab.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# === Training ===
def train(model, X, Y, epochs=10, lr=0.001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        inputs = X.to(device)
        targets = Y.to(device)

        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

# === Text generieren ===
def sample(model, char2idx, idx2char, start_str="hallo", length=100):
    model.eval()
    input_seq = [char2idx.get(ch, 0) for ch in start_str.lower()]
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)  # batch=1

    hidden = None
    generated = start_str

    with torch.no_grad():
        for i in range(length):
            output, hidden = model(input_tensor, hidden)
            probs = torch.softmax(output, dim=-1).cpu()
            char_idx = torch.multinomial(probs, num_samples=1).item()
            char = idx2char.get(char_idx, '?')
            generated += char

            input_tensor = torch.tensor([[char_idx]], dtype=torch.long).to(device)

    return generated

# === Hauptprogramm ===
if __name__ == "__main__":
    seq_length = 50

    if not os.path.exists("training_data_german.txt"):
        print("❌ create the file 'training_data_german.txt' with training data!")
        exit()

    X, Y, char2idx, idx2char = load_text_and_prepare_tensor("training_data_german.txt", seq_length)

    # Modell mit passenden Parametern erstellen
    model = CharRNN(vocab_size=len(char2idx), embedding_dim=256, hidden_dim=256, num_layers=2).to(device)

    # Falls Modell schon existiert, laden
    if os.path.exists("char_rnn_model.pt") and os.path.exists("vocab.json"):
        print("🔁 Loading Modul and Vocabulary...")
        load_model(model, "char_rnn_model.pt")
        char2idx = load_vocab("vocab.json")
        idx2char = {int(i): ch for ch, i in char2idx.items()}
    else:
        print("🧠 New Model Created and will be Trained.")
        save_vocab(char2idx)

    epochen = int(input("for how many epochs do you want to train the AI? "))
    train(model, X, Y, epochs=epochen)

    save_model(model)
    save_vocab(char2idx)

    print("\n✅ Training was succesfull new Model saved and you can now chat with the AI.")

    print("\n🤖 Text with the AI (\"exit\" to Exit):")
    while True:
        user_input = input("you: ")
        if user_input.lower() == "exit":
            break
        output_text = sample(model, char2idx, idx2char, start_str=user_input, length=100)
        print("AI:", output_text)

