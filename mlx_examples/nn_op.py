import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.preprocessing import LabelEncoder

dev = torch.device('mlx')

data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

start = time.time()

X = torch.tensor(X.values, dtype=torch.float32, device=dev)
y = torch.tensor(y, dtype=torch.float32, device=dev).reshape(-1, 1)

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(60, 180, dtype=torch.float32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(180, 1, dtype=torch.float32)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

def model_train(model, X_train, y_train, X_val, y_val):
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 50
    batch_size = 10
    batch_start = torch.arange(0, len(X_train), batch_size)

    best_acc = 0  # init to negative infinity

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
    return best_acc

# create model, train, and get accuracy
model = NN().to(dev)
acc = model_train(model, X, y, X, y)
print("Accuracy: %.2f" % acc)

end = time.time()
print(f"Time: {end - start}s")
