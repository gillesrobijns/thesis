import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import torchvision.models as models

squeezenet = models.squeezenet1_0()

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 64
LR = 0.008         # learning rate
DOWNLOAD_MNIST = True
N_TEST_IMG = 5
MEAN = 0
STDDEV = 0.05

# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,                        # download it if you don't have it
)


def noise(input, mean=0, stddev=0.01):
    input_array = input.numpy()

    noise = np.random.normal(loc=mean, scale=stddev, size=np.shape(input_array))

    out = np.add(input_array, noise)

    output_tensor = torch.from_numpy(out)
    out = output_tensor.float()
    return out
          



print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)


# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 800),
            nn.ReLU(),
            #nn.Linear(800, 800),
            #nn.ReLU(),
            ##nn.Linear(64, 32),
            #nn.Tanh(),
            #nn.Linear(32, 20),   # compress to 20 features
        )
        self.decoder = nn.Sequential(
            #nn.Linear(20, 32),
            #nn.Tanh(),
            ##n.Linear(32, 64),
            ##nn.ReLU(),
            #nn.Linear(800,800),
           # nn.ReLU(),
            nn.Linear(800, 28*28),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

# initialize figure
f, a = plt.subplots(3, N_TEST_IMG, figsize=(5, 2))  # nrows,ncols,
plt.ion()   # continuously plot

# original data (first row) for viewing
view_data = Variable(train_data.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.)
view_noise_data = Variable(noise(view_data.data,MEAN,STDDEV))
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())
    a[1][i].imshow(np.reshape(view_noise_data.data.numpy()[i], (28, 28)), cmap='gray'); a[1][i].set_xticks(()); a[1][i].set_yticks(())

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28*28))   # batch x, shape (batch, 28*28)
        b_y = Variable(x.view(-1, 28*28))   # batch y, shape (batch, 28*28)
        b_label = Variable(y)               # batch label
        b_n = noise(b_x.data,MEAN,STDDEV)   # apply noise to data
        b_n = Variable(b_n)

        encoded, decoded = autoencoder(b_n)

        loss = loss_func(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])

            # plotting decoded image (second row)
            _, decoded_data = autoencoder(view_data)
            for i in range(N_TEST_IMG):
                a[2][i].clear()
                a[2][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[2][i].set_xticks(()); a[2][i].set_yticks(())
                

                
            plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()
