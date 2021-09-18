import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import os
import csv
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

writer = SummaryWriter()


def open_data(datadir1, datadir2):
    """Input:
    datadir1: location of the data
    datadir2: location of the label"""

    dat = pd.read_csv(datadir1, header=None)
    label = pd.read_csv(datadir2, header=None)

    traindat = torch.from_numpy(dat.values)
    trainindex = torch.from_numpy(label.values)

    return traindat, trainindex


def create_encoder_single_conv(in_chs, out_chs, kernel):
    assert kernel % 2 == 1
    return nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=kernel, padding=(kernel - 1) // 2),
                         nn.BatchNorm2d(out_chs),
                         nn.ReLU(inplace=True))


class EncoderInceptionModuleSignle(nn.Module):
    def __init__(self, channels):
        assert channels % 2 == 0
        super().__init__()
        # put bottle-neck layers before convolution
        bn_ch = channels // 2
        # bn_ch = channels
        self.bottleneck = create_encoder_single_conv(channels, bn_ch, 1)
        # bn -> Conv1, 3, 5
        self.conv1 = create_encoder_single_conv(bn_ch, channels, 1)
        self.conv3 = create_encoder_single_conv(bn_ch, channels, 3)
        self.conv5 = create_encoder_single_conv(bn_ch, channels, 5)
        # pool-proj(no-bottle neck)
        self.pool3 = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        # Original inception is concatenation, but use simple addition instead
        bn = self.bottleneck(x)
        out = self.conv1(bn) + self.conv3(bn) + self.conv5(bn) + self.pool3(x)
        return out


class EncoderModule(nn.Module):
    def __init__(self, chs):
        super().__init__()
        layers = [EncoderInceptionModuleSignle(chs)]
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # stages
        self.upch1 = nn.Sequential(nn.Conv2d(1, 4, kernel_size=1),
                                   nn.BatchNorm2d(4),
                                   nn.ReLU(inplace=True))
        # self.upch1 = nn.Conv2d(1, 4, kernel_size=1)     #1*1 ConV bottle neck layer
        self.stage1 = EncoderModule(4)

    def forward(self, x):
        out = self.stage1(self.upch1(x))
        return out.view(out.size(0), -1)


## Decoder
def create_decoder_single_conv(in_chs, out_chs, kernel):
    assert kernel % 2 == 1
    return nn.Sequential(nn.ConvTranspose2d(in_chs, out_chs, kernel_size=kernel, padding=(kernel - 1) // 2),
                         nn.BatchNorm2d(out_chs),
                         nn.ReLU(inplace=True))


class DecoderInceptionModuleSingle(nn.Module):
    def __init__(self, channels):
        assert channels % 2 == 0
        super().__init__()
        # put bottle-neck layers before convolution
        bn_ch = channels // 2
        self.bottleneck = create_decoder_single_conv(channels, bn_ch, 1)
        # bn -> Conv1, 3, 5
        self.conv1 = create_decoder_single_conv(bn_ch, channels, 1)
        self.conv3 = create_decoder_single_conv(bn_ch, channels, 3)
        self.conv5 = create_decoder_single_conv(bn_ch, channels, 5)
        # pool-proj(no-bottle neck)
        self.pool3 = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        # Original inception is concatenation, but use simple addition instead
        bn = self.bottleneck(x)
        out = self.conv1(bn) + self.conv3(bn) + self.conv5(bn) + self.pool3(x)
        return out


class DecoderModule(nn.Module):
    def __init__(self, chs):
        super().__init__()
        layers = [DecoderInceptionModuleSingle(chs)]
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # stages
        self.stage1 = DecoderModule(4)
        self.last = nn.ConvTranspose2d(4, 1, kernel_size=1)
        # self.last = nn.Sequential(nn.ConvTranspose2d(4, 1, kernel_size=1),
        #                    nn.BatchNorm2d(1),
        #                    nn.ReLU(inplace=True))

    def forward(self, x):
        out = x.view(x.size(0), -1, 129, 29)
        out = self.stage1(out)
        return torch.sigmoid(self.last(out))
        # return self.last(out)


class VAE(nn.Module):
    def __init__(self, h_dim=14964, z_dim=50):
        super().__init__()

        # Encoder
        self.encoder = Encoder()
        # Middle
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)
        self.fc_resp = nn.Linear(z_dim, h_dim)
        # Decoder
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        out = self.encoder(x)
        z, mu, logvar = self.bottleneck(out)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc_resp(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        d = self.decode(z)
        return d, mu, logvar


def loss_fn(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # recon_x = recon_x.view(recon_x.size(0), 1 , -1)
    BCE = F.mse_loss(recon_x, x, size_average=False)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.device_count()

    batch_size = 32
    epoch_num = 50
    loss_list = []

    x_train, y_train = open_data('/Users/chenzcha/Desktop/WorkSpace/database/DB_file/exp/spec_nm27.csv',
                                 '/Users/chenzcha/Desktop/WorkSpace/database/DB_file/exp/lable.csv')

    input_data = x_train.view(23491, -1, 129, 29)  # image size
    input_label = y_train

    train_datasets = TensorDataset(input_data, input_label)
    train_loader = DataLoader(dataset=train_datasets,
                              batch_size=batch_size,
                              shuffle=True)

    vaemodel = VAE().to(device)
    # vaemodel.load_state_dict(torch.load('vae.torch', map_location='cpu'))
    optimizer = torch.optim.Adam(vaemodel.parameters(), lr=1e-4)

    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # print(torch.get_num_threads())
    for epoch in range(epoch_num):
        for i, data in enumerate(train_loader):
            inputs, classes = data
            ## if use cuda
            # inputs, classes = inputs.cuda(), classes.cuda()
            inputs = Variable(inputs.float(), requires_grad=True)
            classes = Variable(classes)

            recon_sig, mu, logvar = vaemodel(inputs)
            loss, bce, kld = loss_fn(recon_sig, inputs, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print("### Epoch: ", loss)
            # loss_list.append(loss)

            writer.add_scalar('inception_loss', loss.item(), (epoch + 1) * i)
            writer.add_graph(vaemodel, inputs)

    # torch.save(vaemodel.state_dict(), 'corvaenonnm_827.torch')
    # losssave = pd.DataFrame(data=loss_list)
    # losssave.to_csv("./corvaenonm_827_losslist.csv", sep=',', index=False)
    writer.close()


######## Tsne visualization ########

    N = 23491
    lat = []
    labels = []
    for i in range(N):
        dat = train_loader.dataset[i][0]
        label = train_loader.dataset[i][1]
        latent, m, va = vaemodel.encode(dat.view(1, 1, 129, -1).float().to(device))
        # latent, m, va = vaemodel(dat.view(1, 1, 129, -1).float().to(device))
        lat.append(latent.cpu().detach().numpy())
        labels.append(label.cpu().detach().numpy())

    lat = np.array(lat)
    labels = np.array(labels)
    lat = lat.reshape(N, 50)

    lat.reshape(N, -1)
    constructedsp = pd.DataFrame(data=lat.reshape(N, -1))
    constructedsp.to_csv("./reconssp.csv", sep=',', index=False)
    lat = lat.view(N, 30)

    lat2 = TSNE(n_components=2, learning_rate=19, perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(lat)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cmap = plt.cm.tab20b
    cmaplist = [cmap(i) for i in range(cmap.N)]
    bounds = np.linspace(0, 10, 11)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    scat = ax.scatter(lat2[:, 0], lat[:, 1], c=labels, cmap=cmap, norm=norm)
    cb = plt.colorbar(scat, ticks=[i for i in range(4)])
    cb.set_label('Labels')
    ax.set_title('TSNE plot for VAE Latent Space colour coded by Labels')
    plt.show()

    import matplotlib.pyplot as plt
    plt.plot(loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.show()