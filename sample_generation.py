
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import argparse




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



lb = LabelBinarizer()

def to_categrical(y):
    lb.fit(list(range(1,4)))
    y_one_hot =lb.transform(y)
    y_one_hot=torch.tensor(y_one_hot).type(torch.FloatTensor)
    return y_one_hot


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




if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device =  "mps" if torch.backends.mps.is_available() else "cpu"
    # device =  "mps" if torch.backends.mps.is_available() else "cpu"


    print("=====> Setup model")
    nz=20
    vae = VAE().to(device)
    vae=torch.load(r'ckp/VAE.pth')
    print('loading pretrained model')
    parser = argparse.ArgumentParser()
    Tc_dict={'high':2,'medium':1,'low':0}
    parser.add_argument('--num', type=int, default = 3500)
    parser.add_argument('--Tc', type=str, default='medium',choices=['high','medium','low'])
    args = parser.parse_args()
    print('Input Tc Condition',args.Tc)
    print('Input Num',args.num)

    with torch.no_grad():
        all_f=[]
        for i  in range( args.num):
            label_onehot = torch.zeros((1, 3))
            label_onehot[:,Tc_dict[args.Tc]]=1
            z = torch.randn((1, 20)).to(device)
            output = vae.decoder(z,label_onehot.to(device))
            new_f=get_fumal(output,'new')
            all_f.append(new_f)
            all=pd.DataFrame(all_f)
            all.to_csv('low_fumal.csv')




