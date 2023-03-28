from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import SGD

class MnistClassifier(pl.LightningModule):
    def __init__(self):
        super(MnistClassifier,self).__init__()
        self.fc1 = nn.Linear(28*28,256)
        self.fc2 = nn.Linear(256,128)
        self.out = nn.Linear(128,10)
        self.lr = 0.01
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self,x):
        batch_size, _, _, _ = x.size()
        x = x.view(batch_size,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
    
    def configure_optimizers(self):
        return SGD(self.parameters(),lr = self.lr)
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.loss(logits,y)
        return loss
    
    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        logits = self.forward(x)
        loss = self.loss(logits,y)