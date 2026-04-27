# ===== Part 0: Setup =====
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import requests
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ===== 路径设置（关键）=====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, 'covid.train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'covid.test.csv')
PRED_PATH = os.path.join(BASE_DIR, 'pred.csv')


# ===== Part 1: Download Data =====
def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"{save_path} already exists.")
        return
    print(f"Downloading {save_path}...")
    r = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(r.content)
    print("Done.")


train_url = 'https://drive.google.com/uc?export=download&id=19CCyCgJrUxtvgZF53vnctJiOJ23T5mqF'
test_url = 'https://drive.google.com/uc?export=download&id=1CE240jLm2npU-tdz81-oVKEF3T2yfT1O'

download_file(train_url, TRAIN_PATH)
download_file(test_url, TEST_PATH)


# ===== Part 2: Seed =====
def same_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


same_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)


# ===== Part 3: Feature Selection =====
def select_feat(data):
    state = data[:, :40]
    day1 = data[:, 40:58]
    day2 = data[:, 58:76]

    tp1 = day1[:, -1].reshape(-1, 1)
    tp2 = day2[:, -1].reshape(-1, 1)

    return np.concatenate([state, tp1, tp2], axis=1)


# ===== Part 4: Load & Normalize =====
def load_data(path, train=True, mean=None, std=None):
    data = np.genfromtxt(path, delimiter=',', skip_header=1)

    if train:
        x = data[:, :-1]
        y = data[:, -1]
    else:
        x = data
        y = None

    x = select_feat(x)

    if train:
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True) + 1e-8
        x = (x - mean) / std
        return x, y, mean, std
    else:
        x = (x - mean) / std
        return x, y


x, y, mean, std = load_data(TRAIN_PATH, train=True)
x_test, _ = load_data(TEST_PATH, train=False, mean=mean, std=std)


# ===== Part 5: Split =====
def split(x, y, ratio=0.2):
    idx = np.random.permutation(len(x))
    dev_size = int(len(x) * ratio)
    return x[idx[dev_size:]], x[idx[:dev_size]], y[idx[dev_size:]], y[idx[:dev_size]]


x_train, x_dev, y_train, y_dev = split(x, y)


# ===== Part 6: Dataset =====
class COVIDDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


train_loader = DataLoader(COVIDDataset(x_train, y_train), batch_size=128, shuffle=True)
dev_loader = DataLoader(COVIDDataset(x_dev, y_dev), batch_size=128)
test_loader = DataLoader(COVIDDataset(x_test), batch_size=128)


# ===== Part 7: Model =====
class Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


model = Model(x_train.shape[1]).to(device)


# ===== Part 8: Loss & Optimizer =====
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target) + 1e-8)


criterion = RMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

# ===== Part 9: Training =====
n_epochs = 100
best_loss = float('inf')
stop_count = 0

train_curve, dev_curve = [], []

for epoch in range(n_epochs):
    model.train()
    total = 0

    for x_b, y_b in train_loader:
        x_b, y_b = x_b.to(device), y_b.to(device)

        loss = criterion(model(x_b), y_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    train_loss = total / len(train_loader)
    train_curve.append(train_loss)

    model.eval()
    total = 0
    with torch.no_grad():
        for x_b, y_b in dev_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            total += criterion(model(x_b), y_b).item()

    dev_loss = total / len(dev_loader)
    dev_curve.append(dev_loss)

    scheduler.step(dev_loss)

    print(f"Epoch {epoch + 1}: Train {train_loss:.4f}, Dev {dev_loss:.4f}")

    if dev_loss < best_loss:
        best_loss = dev_loss
        torch.save(model.state_dict(), os.path.join(BASE_DIR, 'best_model.pth'))
        stop_count = 0
    else:
        stop_count += 1

    if stop_count >= 10:
        print("Early stopping")
        break

# ===== Part 10: Plot Learning Curve =====
plt.figure()
plt.plot(train_curve, label='Train')
plt.plot(dev_curve, label='Dev')
plt.legend()
plt.title("Learning Curve")
plt.savefig(os.path.join(BASE_DIR, 'learning_curve.png'))
plt.show()

# ===== Part 11: Prediction vs Ground Truth =====
model.eval()
preds, truths = [], []

with torch.no_grad():
    for x_b, y_b in dev_loader:
        x_b = x_b.to(device)
        pred = model(x_b)

        preds.append(pred.cpu().numpy())
        truths.append(y_b.numpy())

preds = np.concatenate(preds)
truths = np.concatenate(truths)

plt.figure()
plt.scatter(truths, preds, alpha=0.5)
plt.plot([truths.min(), truths.max()], [truths.min(), truths.max()], 'r')
plt.title("Prediction vs Ground Truth")
plt.savefig(os.path.join(BASE_DIR, 'prediction_scatter.png'))
plt.show()

# ===== Part 12: Prediction =====
model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'best_model.pth')))
model.eval()

preds = []
with torch.no_grad():
    for x_b in test_loader:
        x_b = x_b.to(device)
        preds.append(model(x_b).cpu().numpy())

preds = np.concatenate(preds)

with open(PRED_PATH, 'w') as f:
    f.write("id,tested_positive\n")
    for i, v in enumerate(preds):
        f.write(f"{i},{v}\n")

print("All Done!")