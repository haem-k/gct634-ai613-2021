import torch.nn as nn
from torch.nn.modules.conv import Conv2d
import torchaudio
import torch


'''
Multi-label classification
'''

class Baseline(nn.Module):
    def __init__(self,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=96,
                n_class=50):
        """
        Args:
            sample_rate (int): path to load dataset from
            n_fft (int): number of samples for fft
            f_min (float): min freq
            f_max (float): max freq
            n_mels (float): number of mel bin
            n_class (int): number of class
        """
        super(Baseline, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                            n_fft=n_fft,
                                                            f_min=f_min,
                                                            f_max=f_max,
                                                            n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        self.conv0 = nn.Sequential(
            nn.Conv1d(n_mels, out_channels=32, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
            )

        self.conv1 = nn.Sequential(
            nn.Conv1d(32, out_channels=32, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
            )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, out_channels=32, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
            )

        # Aggregate features over temporal dimension.
        self.final_pool = nn.AdaptiveAvgPool1d(1)

        # Predict tag using the aggregated features.
        self.linear = nn.Linear(32, n_class)


    def forward(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)

        x = x.squeeze(1) # for 1D conv
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)       
        x = self.final_pool(x)  

        x = self.linear(x.squeeze(-1)) # after squeeze: [16, 32]
        x = nn.Sigmoid()(x) # for binary cross entropy loss
        return x



class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, pooling=2):
        # To do
        #========================================
        """
        Args:
            input_channels, 
            output_channels, 
            kernel_size, 
            stride, 
            padding, 
            pooling
        """
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
        #========================================

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out



class CNN2D(nn.Module):
    def __init__(self,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=96,
                n_class=50):
        super(CNN2D, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                            n_fft=n_fft,
                                                            f_min=f_min,
                                                            f_max=f_max,
                                                            n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)
        # To do
        #========================================
        self.layer1 = Conv_2d(1, 64, pooling=(4,4))
        self.layer2 = Conv_2d(64, 128, pooling=(3,3))
        self.layer3 = Conv_2d(128, 128, pooling=(3,3))
        self.layer4 = Conv_2d(128, 64, pooling=(2,5))
        #========================================
        self.linear = nn.Linear(64, n_class)
    

    def forward(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = nn.Sigmoid()(x) # for binary cross entropy loss
        return x



class CNN2D_Deep(nn.Module):
    def __init__(self,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=96,
                n_class=50):
        super(CNN2D_Deep, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                            n_fft=n_fft,
                                                            f_min=f_min,
                                                            f_max=f_max,
                                                            n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        self.layer1 = Conv_2d(1, 64)
        self.layer2 = Conv_2d(64, 128)
        self.layer3 = Conv_2d(128, 256)
        self.layer4 = Conv_2d(256, 256)
        self.layer5 = Conv_2d(256, 128)
        self.layer6 = Conv_2d(128, 64)
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(128, n_class)
    

    def forward(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)
        x = self.layer1(x)        # [16, 64, 48, 94]  
        x = self.layer2(x)        # [16, 128, 24, 47]  
        x = self.layer3(x)        # [16, 256, 12, 23]  
        x = self.layer4(x)        # [16, 256, 6, 11]  
        x = self.layer5(x)        # [16, 128, 3, 5]
        x = self.layer6(x)        # [16, 64, 1, 2]
        x = self.dropout(x)
        x = x.view(x.size(0), -1) # [16, 128]
        x = self.linear(x)
        x = nn.Sigmoid()(x) # for binary cross entropy loss
        return x



class CNNTF(nn.Module):
    def __init__(self,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=96,
                n_class=50):
        super(CNNTF, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                            n_fft=n_fft,
                                                            f_min=f_min,
                                                            f_max=f_max,
                                                            n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)        # [16, 1, 96, 188]

        self.freq_0 = Conv_2d(1, 32, kernel_size=(48, 1))
        self.time_0 = Conv_2d(1, 32, kernel_size=(1, 64))

        self.freq_maxpool = nn.MaxPool2d((1, 95))
        self.time_maxpool = nn.MaxPool2d((49, 1))

        self.linear1 = nn.Linear(32*88, n_class)


    def forward(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x) 
        
        x0 = self.freq_0(x) 
        x0 = self.freq_maxpool(x0)

        x1 = self.time_0(x) 
        x1 = self.time_maxpool(x1)

        x0 = x0.squeeze()
        x1 = x1.squeeze()

        x_TF = torch.cat((x0, x1), 2)
        x_TF = x_TF.view(x_TF.size(0), -1)
        x_TF = self.linear1(x_TF)
        x_TF = nn.Sigmoid()(x_TF)

        return x_TF



class CNNTF_Deep(nn.Module):
    def __init__(self,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=96,
                n_class=50):
        super(CNNTF_Deep, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                            n_fft=n_fft,
                                                            f_min=f_min,
                                                            f_max=f_max,
                                                            n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1) 

        self.freq_0 = Conv_2d(1, 32, kernel_size=(48, 1))
        self.freq_1 = Conv_2d(32, 64, kernel_size=(24, 1))

        self.time_0 = Conv_2d(1, 32, kernel_size=(1, 64))
        self.time_1 = Conv_2d(32, 64, kernel_size=(1, 32))

        self.freq_maxpool = nn.MaxPool2d((1, 48))
        self.time_maxpool = nn.MaxPool2d((25, 1))

        self.linear1 = nn.Linear(64*19, 128)
        self.dropout1 = nn.Dropout()
        self.linear2= nn.Linear(128, n_class)
        
    

    def forward(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)
        
        x0 = self.freq_0(x)
        x0 = self.freq_1(x0)

        x1 = self.time_0(x)
        x1 = self.time_1(x1)

        x0 = self.freq_maxpool(x0)
        x1 = self.time_maxpool(x1)

        x0 = x0.squeeze()
        x1 = x1.squeeze()

        x_TF = torch.cat((x0, x1), 2)
        x_TF = x_TF.view(x_TF.size(0), -1)
        x_TF = self.linear1(x_TF)
        x_TF = self.dropout1(x_TF)
        x_TF = self.linear2(x_TF)
        x_TF = nn.Sigmoid()(x_TF)

        return x_TF



class CNNTF2(nn.Module):
    def __init__(self,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=96,
                n_class=50):
        super(CNNTF2, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                            n_fft=n_fft,
                                                            f_min=f_min,
                                                            f_max=f_max,
                                                            n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1) 

        self.freq_0 = Conv_2d(1, 32, kernel_size=(48, 1))
        self.freq_1 = Conv_2d(1, 32, kernel_size=(24, 1))
        self.time_0 = Conv_2d(1, 32, kernel_size=(1, 64))
        self.time_1 = Conv_2d(1, 32, kernel_size=(1, 32))

        self.freq_maxpool = nn.MaxPool2d((1, 95))
        self.time_maxpool = nn.MaxPool2d((49, 1))

        self.linear1 = nn.Linear(32*204, 128)
        self.dropout1 = nn.Dropout()
        self.linear2= nn.Linear(128, n_class)
        

    def forward(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)
        
        x_freq0 = self.freq_0(x) 
        x_freq1 = self.freq_1(x)

        x_time0 = self.time_0(x) 
        x_time1 = self.time_1(x)

        x_freq0 = self.freq_maxpool(x_freq0)
        x_freq1 = self.freq_maxpool(x_freq1)

        x_time0 = self.time_maxpool(x_time0)
        x_time1 = self.time_maxpool(x_time1)

        x_freq0 = x_freq0.squeeze()
        x_freq1 = x_freq1.squeeze()
        x_time0 = x_time0.squeeze()
        x_time1 = x_time1.squeeze()
        
        x_TF = torch.cat((x_freq0, x_freq1, x_time0, x_time1), 2)
        x_TF = x_TF.view(x_TF.size(0), -1)
        x_TF = self.linear1(x_TF)
        x_TF = self.dropout1(x_TF)
        x_TF = self.linear2(x_TF)
        x_TF = nn.Sigmoid()(x_TF)

        return x_TF




