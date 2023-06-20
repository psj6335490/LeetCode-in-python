import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as Data
import os
import torchvision.transforms as T

EPOCH=20
BATCH_SIZE=128
LR=0.001

train_data=torchvision.datasets.MNIST(
    root='./data/MNIST_data',
    transform=T.ToTensor(),
    download=True,
    train=True
)

# print(train_data.data.shape)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.ndf=64
        self.mean_size=3
        self.encoder=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=self.ndf,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(self.ndf),
            nn.ReLU(),

            nn.Conv2d(in_channels=self.ndf, out_channels=2*self.ndf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*self.ndf),
            nn.ReLU(),
        )
        self.fc_mean=nn.Linear(self.ndf*2*7*7,self.mean_size)
        self.fc_log_var=nn.Linear(self.ndf*2*7*7,self.mean_size)

        self.fc_decoder=nn.Linear(self.mean_size,self.ndf*2*7*7)
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.ndf*2,out_channels=self.ndf,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(self.ndf),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=self.ndf, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self,x):
        encode=self.encoder(x)
        out=encode.view(encode.shape[0],-1)
        mean=self.fc_mean(out)
        log_var=self.fc_log_var(out)

        std=torch.exp(0.5*log_var)

        eps=torch.randn_like(std).cuda()
        sample=mean+eps*std

        fc_de=self.fc_decoder(sample)
        fc_de=fc_de.view(fc_de.shape[0],self.ndf*2,7,7)
        decode=self.decoder(fc_de)

        return encode,decode,mean,log_var

    def gen(self,z):
        fc_de = self.fc_decoder(z)
        fc_de = fc_de.view(fc_de.shape[0], self.ndf * 2, 7, 7)
        decode = self.decoder(fc_de)

        return  decode

vae=VAE().cuda()
optim=torch.optim.Adam(vae.parameters(),lr=LR)

train_loader=Data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
from torchvision.utils import save_image
for epoch in range(EPOCH):
    for i ,(bx,by) in enumerate(train_loader):
        bx=bx.cuda()
        encode, decode, mean, log_var=vae(bx)

        rec_loss=torch.nn.functional.binary_cross_entropy(decode,bx,reduction="sum")
        kl_loss=0.5*torch.sum(torch.exp(log_var)+torch.pow(mean,2)-log_var-1)
        loss=rec_loss+kl_loss
        optim.zero_grad()
        loss.backward()
        optim.step()

    print("epoch:", epoch,  " loss:", loss)
    z=torch.zeros(([64,3]))
    v=torch.linspace(start=-0.99,end=0.99,steps=64).view(-1,1)
    z=z+v
    save_path='./saved/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    gens=vae.gen(z.cuda())
    save_image(gens,save_path+"random"+str(epoch+1)+".png")





