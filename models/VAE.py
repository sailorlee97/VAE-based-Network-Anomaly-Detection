import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from models.basic_module import BasicModule
from torchsummary import summary
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F

class VAE(BasicModule):
    def __init__(self,sample_dim:48,rep_dim:48):
        super().__init__()
        self.sample_dim = sample_dim
        self.rep_dim = rep_dim
        self.fc1=nn.Linear(sample_dim,int(0.5*sample_dim))
        self.bn1 = nn.BatchNorm1d(int(0.5*sample_dim))
        self.fc2=nn.Linear(int(0.5*sample_dim),int(0.25*sample_dim))
        self.bn2 = nn.BatchNorm1d(int(0.5*sample_dim))
        self.fcmu=nn.Linear(int(0.25*sample_dim),rep_dim)
        self.fclogvar=nn.Linear(int(0.25*sample_dim),rep_dim)

        self.dropout = nn.Dropout(p = 0.3)

        self.fc3=nn.Linear(rep_dim,int(0.25*sample_dim))
        self.bn3 = nn.BatchNorm1d(int(0.5*sample_dim))
        self.fc4=nn.Linear(int(0.25*sample_dim),int(0.5*sample_dim))
        self.bn4 = nn.BatchNorm1d(int(0.5*sample_dim))
        self.fcxmu=nn.Linear(int(0.5*sample_dim),sample_dim)
        #self.fcxlogvar=nn.Linear(int(0.5*sample_dim),sample_dim)

    def encoder(self, x):
        x = self.fc1(x)
        #x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        #x = self.bn2(x)
        x = F.relu(x)
        #x = x.view(x.size(0), -1)
        mu = self.fcmu(x)
        logvar = self.fclogvar(x)
        return mu,logvar

    def decoder(self, x):
        x=self.fc3(x)
        x = self.dropout(x)
        #x=self.bn3(x)
        x = F.relu(x)
        x=self.fc4(x)
        x = self.dropout(x)
        #x=self.bn4(x)
        x = F.relu(x)
        mu_x=self.fcxmu(x)
        mu_x = torch.sigmoid(mu_x)
        return mu_x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    # def decoder(self, x):
    #     x=self.fc3(x)
    #     #x=self.bn3(x)
    #     x = F.relu(x)
    #     x=self.fc4(x)
    #     #x=self.bn4(x)
    #     x = F.relu(x)
    #     mu_x=self.fcxmu(x)
    #     mu_x = torch.sigmoid(mu_x)
    #     logvar_x=self.fcxlogvar(x)
    #     logvar_x = torch.sigmoid(logvar_x)
    #     return mu_x,  logvar_x
    # 这个是vae的亮点 随机生成隐含向量
    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5*logvar)
    #     eps = torch.randn_like(std)
    #     return eps.mul(std).add_(mu)
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        mu_x=self.decoder(z)
        #x_recon=self.reparameterize(mu_x, logvar_x)
        return mu_x,z,mu, logvar


# reconstruction_function = nn.MSELoss(size_average=False)
# def loss_function(recon_x, x, mu, logvar):
#     """
#     recon_x: generating images
#     x: origin images
#     mu: latent mean
#     logvar: latent log variance
#     """
#     BCE = reconstruction_function(recon_x, x)  # mse loss
#     # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
#     KLD = torch.sum(KLD_element).mul_(-0.5)
#     # KL divergence
#     return BCE + KLD


if __name__ == '__main__':
    # input = torch.randn(32, 48)

    data = pd.read_csv('../unsw/Normal.csv')
    data = (data.astype(np.float32) - 127.5) / 127.5
    X_normal = data.values.reshape(data.shape[0], 44)  # 变成矩阵格式
    print(X_normal.shape)
    X_normal = torch.FloatTensor(X_normal)
    X_normal = TensorDataset(X_normal)  # 对tensor进行打包
    train_loader = DataLoader(dataset=X_normal, batch_size=32,shuffle=True)  # 数据集放入Data.DataLoader中，可以生成一个迭代器，从而我们可以方便的进行批处理

    vae = VAE(44, 44)
    optimizer = torch.optim.Adam(vae.parameters(), 0.0008)

    loss_func = nn.MSELoss(reduction='mean')
    summary(vae,(32, 44))
    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0
        for step, (x,) in enumerate(train_loader):
            x_recon,z,mu, logvar= vae.forward(x)
            print(x_recon)
            reconst_loss = F.binary_cross_entropy(x_recon, x, size_average=False)
            kl_div = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            # 反向传播和优化
            loss = reconst_loss + kl_div
            # loss = calculate_losses(x_recon,x)
            # x_recon = torch.FloatTensor(x_recon)
            # loss = F.kl_div(x_recon, x,reduction='mean')
            # loss = loss_func(x_recon, x)
            optimizer.zero_grad()
            # 计算中间的叶子节点，计算图
            loss.backward()
            # 内容信息反馈
            optimizer.step()

            total_loss += loss.item() * len(x)
    # input = input.permute(0, 2, 1)
        print('Epoch {}/{} : loss: {:.4f}'.format(
            epoch + 1, num_epochs, loss))
    out = vae(input)

