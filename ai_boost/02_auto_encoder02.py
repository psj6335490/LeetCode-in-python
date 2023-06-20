import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torchvision.transforms as T
import torch.nn as nn


transform=T.Compose([
    T.RandomCrop(32,padding=4),
    T.RandomRotation(90),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
])

train_data=torchvision.datasets.CIFAR10(
    root="./data/cifar_data",
    train=True,
    transform=transform,
    download=True,
)

# for x,y in train_loader:
#     plt.imshow(x[0].permute(1,2,0))
#     plt.show()
#     pass


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        ndf=64
        self.encoder=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=ndf,kernel_size=4,padding=1,stride=2),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=ndf, out_channels=2*ndf, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(2*ndf),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=2*ndf, out_channels=4 * ndf, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(4 * ndf),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(in_channels=4*ndf,out_channels=2*ndf,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(2 * ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(in_channels=2 * ndf, out_channels= ndf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1 * ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(in_channels=ndf, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh(),
        )

    def forward(self,x):
        encoder=self.encoder(x)
        decoder=self.decoder(encoder)
        return encoder,decoder


EPOCH=10
BATCH_SIZE=128
LR=0.005
autoencoder=AutoEncoder()
if torch.cuda.is_available():
    autoencoder=autoencoder.cuda()
train_loader=Data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
lamda=0.01

optim=torch.optim.Adam(autoencoder.parameters(),lr=LR)
loss_func=nn.MSELoss()

N_test_image=5
f,a=plt.subplots(2,N_test_image,figsize=(5,2))
plt.ion()


for epoch in range(EPOCH):
    for i,(x,y) in enumerate(train_loader):
        b_x=x.cuda()
        encoder,decoder=autoencoder(b_x)
        reg_loss=0
        for p in autoencoder.parameters():
            reg_loss+=torch.sum(abs(p))

        # loss=loss_func(decoder,b_x)+lamda*reg_loss
        loss = loss_func(decoder, b_x)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % 499==0:
            print("epoch:",epoch," i:",i," loss:",loss)
            _, decoder = autoencoder(b_x)
            decoder=decoder.data.cpu().numpy().transpose([0,2,3,1])
            decoder=(decoder+1)/2

            b_x = b_x.data.cpu().numpy().transpose([0, 2, 3, 1])
            b_x = (b_x + 1) / 2

            for j in range(N_test_image):

                a[0][j].clear()
                a[0][j].imshow(b_x[j])
                a[0][j].set_xticks(())
                a[0][j].set_yticks(())

                a[1][j].clear()
                a[1][j].imshow(decoder[j])
                a[1][j].set_xticks(())
                a[1][j].set_yticks(())

            plt.draw()
            plt.pause(0.05)


plt.ioff()
plt.show()