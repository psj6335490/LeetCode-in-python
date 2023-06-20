import matplotlib.pyplot as plt
import numpy as np
import torch
import  torchvision
import torchvision.transforms as  transform
import torch.nn as nn
import torch.utils.data as data

train_data=torchvision.datasets.MNIST(
    root="./data/MNIST_data",
    transform=transform.ToTensor(),
    train=True,
    download=True
)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(28*28,3),
            nn.Tanh(),
        )

        self.decoder=nn.Sequential(
            nn.Linear(3,28*28),
            nn.ReLU(),
        )

    def forward(self,x):
        encoder=self.encoder(x)
        decoder=self.decoder(encoder)
        return encoder,decoder

EPOCH=10
BATCH_SIZE=64
LR=0.005
autoencoder=AutoEncoder()
if torch.cuda.is_available():
    autoencoder=autoencoder.cuda()

optim=torch.optim.Adam(autoencoder.parameters(),lr=LR)

loss_func=nn.MSELoss()

N_test_image=5
# f,a=plt.subplots(2,N_test_image,figsize=(5,2))
# plt.ion()

# view_data=train_data.data[:N_test_image].view(-1,28*28).type(torch.FloatTensor)
#
# for i in range(N_test_image):
#     a[0][i].imshow(np.reshape(view_data.data.numpy()[i],(28,28)),cmap="gray")
#     a[0][i].set_xticks(())
#     a[0][i].set_yticks(())


train_loader=data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)

for epoch in range(EPOCH):
    for i,(x,y) in enumerate(train_loader):
        if torch.cuda.is_available():
            x=x.cuda()
        b_x=x.view(-1,28*28)
        encoder,decoder=autoencoder(b_x)
        loss=loss_func(decoder,b_x)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % 499==0:
            print("epoch:",epoch," i:",i," loss:",loss)
#             _, decoder = autoencoder(view_data.cuda())
#
#             for i in range(N_test_image):
#                 a[1][i].clear()
#                 a[1][i].imshow(np.reshape(decoder.cpu().data.numpy()[i], (28, 28)), cmap="gray")
#                 a[1][i].set_xticks(())
#                 a[1][i].set_yticks(())
#             plt.draw()
#             plt.pause(0.05)
#
#
# plt.ioff()
# plt.show()

from mpl_toolkits.mplot3d import Axes3D
from  matplotlib import cm
view_data=train_data.data[:200].view(-1,28*28).type(torch.FloatTensor)/255.
encoder,_=autoencoder(view_data.cuda())
fig=plt.figure()
ax=Axes3D(fig)
X=encoder.cpu().data[:,0].numpy()
Y=encoder.cpu().data[:,1].numpy()
Z=encoder.cpu().data[:,2].numpy()

values=train_data.targets[:200].numpy()

for x,y,z,s in zip(X,Y,Z,values):
    c=cm.rainbow(int(255*s/9))
    ax.text(x,y,z,s,backgroundcolor=c)
ax.set_xlim(X.min(),X.max())
ax.set_ylim(Y.min(),Y.max())
ax.set_zlim(Z.min(),Z.max())

plt.show()
plt.pause(50)

# if __name__ == '__main__':
#     pass
    # print(train_data.data.size())
    # print(train_data.targets.size())
    # import matplotlib.pyplot as plt
    # plt.imshow(train_data.data[0].numpy(),cmap="gray")
    # plt.title(train_data.targets[0].numpy())
    # plt.show()

