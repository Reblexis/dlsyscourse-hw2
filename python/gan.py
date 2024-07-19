import needle as ndl
import numpy as np
from needle import nn
from matplotlib import pyplot as plt

A = np.array([[1, 2], [-0.2, 0.5]])
mu = np.array([2, 1])
num_samples = 3200

data = np.random.normal(0, 1, (num_samples, 2)) @ A + mu

model_G = nn.Linear(2, 2)


def sample_G(model_G, num_samples):
    Z = ndl.Tensor(np.random.normal(0, 1, (num_samples, 2)))
    return model_G(Z).numpy()


data_fake_init = sample_G(model_G, 2000)



model_D = nn.Sequential(
    nn.Linear(2, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 2),
)

loss_D = nn.SoftmaxLoss()

opt_G = ndl.optim.Adam(model_G.parameters(), lr=0.01)


def update_G(Z, model_G, model_D, loss_D, opt_G):
    opt_G.reset_grad()
    X_fake = model_G(Z)
    Y_fake = model_D(X_fake)
    batch_size = Z.shape[0]

    ones = ndl.ones(batch_size, dtype='int32')
    loss = loss_D(Y_fake, ones)
    loss.backward()
    opt_G.step()

opt_D = ndl.optim.Adam(model_D.parameters(), lr=0.01)

def update_D(Z, X, model_G, model_D, loss_D, opt_D):
    opt_D.reset_grad()
    X_fake = model_G(Z)
    Y_fake = model_D(X_fake)
    Y_real = model_D(X)
    batch_size = Z.shape[0]
    ones = ndl.ones(batch_size, dtype='int32')
    zeros = ndl.zeros(batch_size, dtype='int32')

    loss = loss_D(Y_real, ones) + loss_D(Y_fake, zeros)
    loss.backward()
    opt_D.step()


def train_gan(data, batch_size, num_epochs):
    assert data.shape[0] % batch_size == 0

    for epoch in range(num_epochs):
        begin = (batch_size * epoch) % data.shape[0]
        X = ndl.Tensor(data[begin:begin + batch_size])
        Z = ndl.Tensor(np.random.normal(0, 1, (batch_size, 2)))

        update_G(Z, model_G, model_D, loss_D, opt_G)
        update_D(Z, X, model_G, model_D, loss_D, opt_D)

train_gan(data, 32, 2000)


fake_data_trained = sample_G(model_G, 100)

plt.scatter(data[:, 0], data[:, 1], color='blue', label='real data')
plt.scatter(data_fake_init[:, 0], data_fake_init[:, 1], color='pink', label='G(z) init')
plt.scatter(fake_data_trained[:, 0], fake_data_trained[:, 1], color='red', label='G(z) trained')
plt.legend()

plt.show()


gA, gmu = model_G.parameters()

print(A.T @ A)

gA = gA.numpy()

print(gA.T @ gA)

class GANLoss(nn.Module):
    def __init__(self, model_D, opt_D):
        self.model_D = model_D
        self.opt_D = opt_D
        self.loss_D = nn.SoftmaxLoss()

    def _update_D(self, X_fake, X_real):
        self.opt_D.reset_grad()
        Y_fake = self.model_D(X_fake.detach())
        Y_real = self.model_D(X_real)
        batch_size = X_fake.shape[0]
        ones = ndl.ones(batch_size, dtype='int32')
        zeros = ndl.zeros(batch_size, dtype='int32')
        loss = self.loss_D(Y_real, ones) + self.loss_D(Y_fake, zeros)
        loss.backward()
        self.opt_D.step()

    def forward(self, X_fake, X_real):
        self._update_D(X_fake, X_real)

        Y_fake = self.model_D(X_fake)
        batch_size = X_fake.shape[0]
        ones = ndl.ones(batch_size, dtype='int32')
        loss = self.loss_D(Y_fake, ones)
        return loss

model_G = nn.Sequential(nn.Linear(2, 2))
opt_G = ndl.optim.Adam(model_G.parameters(), lr=0.01)

model_D = nn.Sequential(
    nn.Linear(2, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 2),
)

opt_D = ndl.optim.Adam(model_D.parameters(), lr=0.01)

gan_loss = GANLoss(model_D, opt_D)

def train_gan(data, batch_size, num_epochs):
    assert data.shape[0] % batch_size == 0

    for epoch in range(num_epochs):
        opt_G.reset_grad()

        begin = (batch_size * epoch) % data.shape[0]
        X = ndl.Tensor(data[begin:begin + batch_size, :])
        Z = ndl.Tensor(np.random.normal(0, 1, (batch_size, 2)))

        X_fake = model_G(Z)
        loss = gan_loss(X_fake, X)
        loss.backward()
        opt_G.step()


train_gan(data, 32, 2000)

fake_data_trained = sample_G(model_G, 100)

plt.scatter(data[:, 0], data[:, 1], color='blue', label='real data')
plt.scatter(data_fake_init[:, 0], data_fake_init[:, 1], color='pink', label='G(z) init')
plt.scatter(fake_data_trained[:, 0], fake_data_trained[:, 1], color='red', label='G(z) trained')
plt.legend()
plt.show()