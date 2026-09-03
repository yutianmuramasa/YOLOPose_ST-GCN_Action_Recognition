import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import ActionDataset
from models.stgcn_model import STGCN

# ======================
# 配置
# ======================
DATASET_DIR = "dataset2"
BATCH_SIZE = 8
EPOCHS = 50
LR = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# 数据
# ======================
dataset = ActionDataset(DATASET_DIR)
num_class = len(dataset.class_names)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

print("Classes:", dataset.class_names)

# ======================
# 模型
# ======================
model = STGCN(num_class=num_class).to(DEVICE)
class_weights = torch.tensor([
    1.0,  # background
    2.5,  # drink
    2.5,  # raisehand
    2.5   # zhangbi
]).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ======================
# 训练
# ======================
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 验证
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total if total > 0 else 0
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss {total_loss:.3f} | Val Acc {acc:.3f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "weights/stgcn_best.pth")
        print("✅ Best model saved")

print("🎉 Training Finished")
