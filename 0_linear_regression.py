import torch
from torch.nn import Linear, MSELoss
from torch.optim import SGD

LEARNING_RATE = 0.01
N_ITERS = 10000
N_SAMPLES = 200
N_INPUT_DIMS = 20
N_OUTPUT_DIMS = 10

x = torch.randn((N_SAMPLES, N_INPUT_DIMS))
y = torch.randn((N_SAMPLES, N_OUTPUT_DIMS))

model = Linear(N_INPUT_DIMS, N_OUTPUT_DIMS)
criterion = MSELoss()
opt = SGD(model.parameters(), lr=LEARNING_RATE)

for i in range(N_ITERS):
    model.zero_grad()
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    opt.step()
    print("Iteration {}/{} Loss: {:.2f}".format(i + 1, N_ITERS, loss.item()), end="\r")
