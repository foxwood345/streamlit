import torch.nn as nn
import torch.nn.functional as F
import torch

from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

class BasicModel(nn.Module):
  def __init__(self):
    super(BasicModel, self).__init__()
    self.conv1 = nn.Conv2d(1, 20, 5, 1)
    self.conv2 = nn.Conv2d(20, 50, 5, 1)
    self.fc1 = nn.Linear(4*4*50, 500)
    self.fc2 = nn.Linear(500, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 4*4*50)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

# class BasicModel(LightningModule):
#   def __init__(self):
#     super(BasicModel, self).__init__()
#     self.conv1 = nn.Conv2d(1, 20, 5, 1)
#     self.conv2 = nn.Conv2d(20, 50, 5, 1)
#     self.fc1 = nn.Linear(4*4*50, 500)
#     self.fc2 = nn.Linear(500, 10)

#   def forward(self, x):
#     x = F.relu(self.conv1(x))
#     x = F.max_pool2d(x, 2, 2)
#     x = F.relu(self.conv2(x))
#     x = F.max_pool2d(x, 2, 2)
#     x = x.view(-1, 4*4*50)
#     x = F.relu(self.fc1(x))
#     x = self.fc2(x)
#     return F.log_softmax(x, dim=1)

#   def training_step(self, batch, batch_idx):
#     x, y = batch
#     pred = self(x)
#     loss = F.nll_loss(pred, y)
#     acc = accuracy(pred, y)
#     self.log("train_loss_epoch", loss, on_epoch=True, on_step=False)
#     self.log("train_loss_step", loss)
#     self.log("acc", acc, on_epoch=True, on_step=False)
#     return {"loss": loss, "acc" : acc}

#   def configure_optimizers(self):
#     return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.5)

  