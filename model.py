import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
import os
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter()
lb = LabelBinarizer()


def to_categrical(y):
    lb.fit(list(range(1,4)))
    y_one_hot =lb.transform(y)
    y_one_hot=torch.tensor(y_one_hot).type(torch.FloatTensor)
    return y_one_hot


def get_label(x):
    spaces =[  0, 40, 77,150 ]
    index = np.digitize(x, spaces)
    return int(index),x


class Data_base():
    def __init__(self):
        all_data=pd.read_csv('data/element.csv',)
        print(np.shape(all_data))
        Tc_data = all_data.iloc[1:,-1]
        self.Tc_data=[get_label(i) for i in np.array(Tc_data)]
        self.all_data=np.array(all_data.iloc[1:,1:87])
    def __getitem__(self, item):

        return self.all_data[item],np.array(self.Tc_data[item])

    def __len__(self):
        return len(self.all_data)

species1=['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti', 'V', 
            'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 
            'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 
            'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 
            'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu']


def get_fumal(array,label):
    gen_im=array.detach().cpu()
    gen_im=gen_im.view(-1,86)
    array=np.array(gen_im[0])
    f=str()
    for i,j in enumerate(array):
        if float(str(j))>0:
            f=f+str(species1[i])+str(j)[:4]
    print(str(label),':',f)
    return f




class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1,16,kernel_size=3,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),     #(89+2*1-3)/2+1=43

            nn.Conv1d(16,32,kernel_size=3,stride=2,padding=1), 
            nn.LeakyReLU(0.2,inplace=True),             #(43+2*1-3)/2+1=22

            nn.Conv1d(32,32,kernel_size=3,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),           #(22+2*1-3)/2+1=12
        )
        self.encoder_fc1=nn.Linear(32*12,nz)
        self.encoder_fc2=nn.Linear(32*12,nz)
        self.decoder_fc = nn.Linear(nz+3,32 * 12)
        self.decoder_deconv = nn.Sequential(   
            nn.ConvTranspose1d(32, 16, 2, 2, 1),#2*（12-1）+2-2*1=22
            nn.ReLU(inplace=True),              
            nn.ConvTranspose1d(16, 16, 3, 2, 1),#2*（22-1）+3-2*1=43
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, 1, 2, 2, 0),#2*（43-1）+2-2*0=86
            nn.ReLU(inplace=True),
        )


    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).to(device)
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x,label):
        z ,mean.logstd= self.encoder(x,label)
        output = self.decoder(z,label)
        return output
    def encoder(self,x,label):#[b,1,86] [b,1,3] =[b,1,89]   
        label=torch.unsqueeze(label,1)
        x=torch.cat([x,label], axis=2)
        out1, out2 = self.encoder_conv(x), self.encoder_conv(x)
        # print(out1.size())
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))
        z = self.noise_reparameterize(mean, logstd)
        return z,mean,logstd
    def decoder(self,z,label):
        z=torch.cat([z,label], axis=1)
        out3 = self.decoder_fc(z)
        out3=  nn.ReLU()(out3)
        out3 = out3.view(out3.shape[0], 32, -1)
        out3 = self.decoder_deconv(out3)
        return out3


class Discriminator(nn.Module):#W-GAN-GP。
    def __init__(self,outputn=1):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv1d(1, 32, 3, stride=1, padding=1),#(86+2*1-3)/1+1=86
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(32, 64, 3, stride=2, padding=1),#(86+2*1-3)/2+1=43
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(64, 64, 3, stride=2, padding=1),#(43+2*1-3)/2+1=22
            nn.LeakyReLU(0.2, True),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*22, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, outputn),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.dis(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)

def loss_function(recon_x,x,mean,logstd):
    MSE = MSECriterion(recon_x,x)
    var = torch.pow(torch.exp(logstd),2)
    KLD = -0.5 * torch.sum(1+torch.log(var)-torch.pow(mean,2)-var)
    return MSE+KLD




import warnings

warnings.filterwarnings("ignore") 

if __name__ == '__main__':

    nz=20
    nepoch=30000
    D_iter=6
    lr=1e-4
    batch_size=10000
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device =  "mps" if torch.backends.mps.is_available() else "cpu"
    print(device)
    

    cudnn.benchmark = True
    data_set=Data_base()
    data_loader=DataLoader(data_set,batch_size=batch_size,shuffle=True,num_workers=0)
    print("=====> Setup VAE")
    vae = VAE().to(device)
    
    print("=====> Setup D")
    D = Discriminator(1).to(device)
    print("=====> Setup C")
    C = Discriminator(3).to(device)

    criterion = nn.BCELoss().to(device)
    MSECriterion = nn.MSELoss().to(device)

    print("=====> Setup optimizer")
    optimizerD = optim.Adam(D.parameters(), lr=lr,)
    optimizerC = optim.Adam(C.parameters(), lr= lr,)
    optimizerVAE = optim.Adam(vae.parameters(),  lr=lr,)


    
    for epoch in range(nepoch):
        # ajust learning rate
        if epoch %100== 0:
            optimizerD.param_groups[0]['lr'] *= 0.9
            optimizerC.param_groups[0]['lr'] *= 0.9
            optimizerVAE.param_groups[0]['lr'] *= 0.9
        for i, (data,label) in enumerate(data_loader, 0):
            

            data=data.type(torch.FloatTensor).to(device).unsqueeze(1)
            label_onehot = to_categrical(label[:,0]).to(device)
            batch_size = data.shape[0]
            print(batch_size)
  

            # training the discriminator and Tc classifier
            if i%D_iter == 0:
        
                output = C(data)
                real_label = label_onehot.to(device)  # real supcon labeled 
                errC = criterion(output, real_label)
                C.zero_grad()
                errC.backward()
                optimizerC.step()

                output = D(data)
                real_label = torch.ones(batch_size).to(device)   # real data labeled as 1
                fake_label = torch.zeros(batch_size).to(device)  # fake data labeled as 0
                errD_real = criterion(output, real_label)
                z = torch.randn(batch_size, nz).to(device)


                fake_data = vae.decoder(z,label_onehot.to(device))
                output = D(fake_data)
                errD_fake = criterion(output, fake_label)
                
                errD = (errD_real+errD_fake)
                D.zero_grad()
                errD.backward()
                optimizerD.step()

            # update VAE(G)1
            z,mean,logstd = vae.encoder(data,label_onehot.to(device))
            
            recon_data = vae.decoder(z,label_onehot.to(device))
            vae_loss1 = loss_function(recon_data,data,mean,logstd)

            # update VAE(G)2
            output = D(recon_data)
            real_label = torch.ones(batch_size).to(device)
            vae_loss2 = criterion(output,real_label)

            # update VAE(G)3
            output = C(recon_data)
            real_label = label_onehot.to(device)
            vae_loss3 = criterion(output, real_label)
            vae.zero_grad()
            vae_loss = vae_loss1+vae_loss2+vae_loss3
            vae_loss.backward()
            optimizerVAE.step()

        
        writer.add_scalar('Loss_D', errD, epoch)
        writer.add_scalar('Loss_C', errC, epoch)
        writer.add_scalar('Loss_G', vae_loss, epoch)



        if epoch%10==0 :
            print('[%d/%d] Loss_D: %.4f Loss_C: %.4f Loss_G: %.4f'
                % (epoch, nepoch, 
                    errD.item(),errC.item(),vae_loss.item()),vae_loss1.item())


            real_label=label_onehot
            get_fumal(data.cpu(),'input')
            sample = torch.randn(data.shape[0], nz).to(device)
            print(torch.argmax(real_label[0]))
            output = vae.decoder(sample,real_label)
            tc=C(output)
            print(torch.argmax(tc.data[0]))
            get_fumal(output.cpu(),'new')

        if epoch%100==1:
            torch.save(vae, 'VAE.pth')
            torch.save(D,'Discriminator.pth')
            torch.save(C,'Classifier.pth')

