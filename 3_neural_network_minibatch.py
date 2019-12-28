import math
import torch
from torch.nn import Sequential, Sigmoid, Linear, CrossEntropyLoss
from torch.optim import SGD

LEARNING_RATE = 0.01
N_EPOCHS = 100
N_BATCH_SIZE = 32000
N_SAMPLES = 1000000
N_INPUT_DIMS = 20
N_HIDDEN_DIMS = 128
N_OUTPUT_DIMS = 10

x = torch.randn((N_SAMPLES, N_INPUT_DIMS))
y = torch.randn((N_SAMPLES, N_OUTPUT_DIMS)).max(dim=1)[1]

model = Sequential(
    Linear(N_INPUT_DIMS, N_HIDDEN_DIMS),
    Sigmoid(),
    Linear(N_HIDDEN_DIMS, N_OUTPUT_DIMS),
)
criterion = CrossEntropyLoss()
opt = SGD(model.parameters(), lr=LEARNING_RATE)

batch_count = N_SAMPLES // N_BATCH_SIZE
for i in range(N_EPOCHS):
    avg_acc = 0
    avg_loss = 0
    for j in range(batch_count):
        batch_x = x[j * N_BATCH_SIZE: (j + 1) * N_BATCH_SIZE]
        batch_y = y[j * N_BATCH_SIZE: (j + 1) * N_BATCH_SIZE]
        model.zero_grad()
        y_hat = model(batch_x)
        loss = criterion(y_hat, batch_y)
        loss.backward()
        acc = torch.eq(y_hat.max(dim=1)[1], batch_y).float().mean()
        opt.step()
        avg_loss = (loss.item() + j * avg_loss) / (j + 1)
        avg_acc = (acc.item() + j * avg_acc) / (j + 1)
        print("Epoch: {}/{} Batch: {}/{} Loss: {:.2f} Accuracy: {:.2%}".format(i + 1, N_EPOCHS, j + 1, batch_count, avg_loss, avg_acc), end="\r")
    print()
