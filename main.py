import torch
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-np.pi, np.pi, 400)
y = np.sin(x)
x = x / np.pi
x_tensor = torch.from_numpy(x).float().reshape(-1,1)
y_tensor = torch.from_numpy(y).float().reshape(-1,1)
model = torch.nn.Sequential(
    torch.nn.Linear(1, 60),
    torch.nn.ReLU(),
    torch.nn.Linear(60, 25),
    torch.nn.ReLU(),
    torch.nn.Linear(25, 1)
)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
fig, axs = plt.subplots(3, 3, figsize=(9, 9))
axs = axs.ravel()
mae_list = []
ep=75
for i in range(9):
    for epoch in range(ep):
        y_pred = model(x_tensor)
        loss = criterion(y_pred, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mae = torch.mean(torch.abs(y_pred - y_tensor))
        mae_list.append(mae)
    axs[i].plot(x,y, label ='True')
    axs[i].plot(x,y_pred.detach().numpy(), label= 'predicted')
    axs[i].set_title("after {:} Epoch MAE ={:.4f}".format(ep,mae),fontsize=9 )
    ep+=75
    axs[i].legend()
plt.savefig('plot.pdf')
plt.show()

m = torch.jit.script (model); m.save('model.torch')
with open('mae.txt', 'w') as f:
    f.write(str(float((mae_list[-1]))))

