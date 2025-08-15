import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):

    def __init__(self, size):
        super(Chomp1d,self).__init__()
        self.chomp_size = size

    def forward(self, x):
        return x[:,:,:-self.chomp_size]
    

class TemporalConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super(TemporalConvBlock, self).__init__()

        self.padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=self.padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(self.padding)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=self.padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(self.padding)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.casual = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.upordownsampling = nn.Conv1d(in_channels, out_channels, kernel_size = 1) if in_channels != out_channels else None
    
    def forward(self, x):
        out = self.casual(x)
        if self.upordownsampling is not None:
            x = self.upordownsampling(x)

        return x+out


class TCN(nn.Module):

    def __init__(self, in_channels, channels, out_channels, kernel_size):
        super(TCN, self).__init__()

        num_channel = len(channels)
        dilation = 1

        layer = []

        for i in range(num_channel):
            in_ = in_channels if i == 0 else channels[i-1]
            out_ = channels[i]
            layer.append(
                TemporalConvBlock(in_, out_, kernel_size=kernel_size, dilation=dilation)
            )
            dilation *= 2

        layer.append(
            TemporalConvBlock(channels[-1], out_channels, kernel_size=kernel_size, dilation=dilation)
        )

        self.pooling = torch.nn.AdaptiveMaxPool1d(1)

        self.network = nn.Sequential(*layer)
    
    def forward(self, x):
        out = self.network(x)
        out = self.pooling(out)
        return out.squeeze(2)
    



