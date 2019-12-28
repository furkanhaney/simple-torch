import torch
from torch.nn import Sequential, Sigmoid, Linear, CrossEntropyLoss
from torch.optim import SGD

LEARNING_RATE = 0.01
N_ITERS = 10000
N_SAMPLES = 200
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

for i in range(N_ITERS):
    model.zero_grad()
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    acc = torch.eq(y_hat.max(dim=1)[1], y).float().mean().item()
    opt.step()
    print("Iteration {}/{} Loss: {:.2f} Accuracy: {:.2%}".format(i + 1, N_ITERS, loss.item(), acc), end="\r")
