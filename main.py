import torch
from models import KanNet3, make_model_image
from dataset import train_loader, test_loader
from utils import train

import matplotlib.pyplot as plt

plt.style.use("ggplot")

device = torch.device("cuda:0")
model = KanNet3().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_of_epochs = 100

train_losses, valid_losses, train_accuracy, valid_accuracy, time_results = train(
    model, optimizer, train_loader, test_loader, "cross_entropy", num_of_epochs, device
)


plt.plot(range(num_of_epochs), train_losses, label="train")
plt.plot(range(num_of_epochs), valid_losses, label="valid")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("losses.png")
plt.close("all")

plt.plot(range(num_of_epochs), train_accuracy, label="train")
plt.plot(range(num_of_epochs), valid_accuracy, label="valid")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accs.png")
plt.close("all")

plt.plot(range(num_of_epochs), time_results["plot"])
plt.xlabel("Epoch")
plt.ylabel("$time, sec$")
plt.savefig("time.png")
plt.close("all")

print(f"mean time by epoch: {time_results['mean_epoch']}")
print(f"total time : {time_results['total']}")

make_model_image(model, train_loader, device, "model")
