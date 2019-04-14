import torch as T
from torch import nn
from torch.nn import functional as F
from .layer import GraphConv, GraphPool, GraphOutput
import numpy as np
from torch import optim
import time
from .util import dev


class QSAR(nn.Module):
    def __init__(self, hid_dim, n_class):
        super(QSAR, self).__init__()
        self.gcn1 = GraphConv(input_dim=43, conv_width=128)
        self.gcn2 = GraphConv(input_dim=134, conv_width=128)
        self.gop = GraphOutput(input_dim=134, output_dim=128)
        # self.bn = nn.BatchNorm2d(80)
        self.pool = GraphPool()
        self.fc2 = nn.Linear(hid_dim, n_class)

    def forward(self, atoms, bonds, edges):
        atoms = self.gcn1(atoms, bonds, edges)
        # atoms = self.bn(atoms)
        atoms = self.pool(atoms, edges)
        atoms = self.gcn2(atoms, bonds, edges)
        # atoms = self.bn(atoms)
        atoms = self.pool(atoms, edges)
        fp = self.gop(atoms, bonds, edges)
        out = F.sigmoid(self.fc2(fp))
        return out

    def fit(self, loader_train, loader_valid, path, epochs=1000, early_stop=100, lr=1e-3):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        best_loss = np.inf
        last_saving = 0
        for epoch in range(epochs):
            t0 = time.time()
            for Ab, Bb, Eb, yb in loader_train:
                Ab, Bb, Eb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), yb.to(dev)
                optimizer.zero_grad()
                y_ = self.forward(Ab, Bb, Eb)
                ix = yb == yb
                yb, y_ = yb[ix], y_[ix]
                loss = criterion(y_, yb)
                loss.backward()
                optimizer.step()
            loss_valid = self.evaluate(loader_valid)
            print('[Epoch:%d/%d] %.1fs loss_train: %f loss_valid: %f' % (
                epoch, epochs, time.time() - t0, loss.item(), loss_valid))
            if loss_valid < best_loss:
                T.save(self, path + '.pkg')
                print('[Performance] loss_valid is improved from %f to %f, Save model to %s' % (
                    best_loss, loss_valid, path + '.pkg'))
                best_loss = loss_valid
                last_saving = epoch
            else:
                print('[Performance] loss_valid is not improved.')
            if early_stop is not None and epoch - last_saving > early_stop: break
        return T.load(path + '.pkg')

    def evaluate(self, loader):
        loss = 0
        criterion = nn.BCELoss()
        for Ab, Bb, Eb, yb in loader:
            Ab, Bb, Eb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), yb.to(dev)
            y_ = self.forward(Ab, Bb, Eb)
            ix = yb == yb
            yb, y_ = yb[ix], y_[ix]
            loss += criterion(y_, yb).item()
        return loss / len(loader)

    def predict(self, loader):
        score = []
        for Ab, Bb, Eb, yb in loader:
            Ab, Bb, Eb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), yb.to(dev)
            y_ = self.forward(Ab, Bb, Eb).data.cpu()
            ix = yb == yb
            yb, y_ = yb[ix], y_[ix]
            score.append(y_.detach().cpu())
        score = T.cat(score, dim=0).numpy()
        return score
