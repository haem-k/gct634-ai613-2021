import torch.nn as nn
import torchaudio
import torch



'''
Metric Learning
'''
class LinearProjection(nn.Module):
    """
    Backbone model for linear projection
    mel spectrogam to embedding
    """
    def __init__(self,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=96):
        """
        Args:
            sample_rate (int): path to load dataset from
            n_fft (int): number of samples for fft
            f_min (float): min freq
            f_max (float): max freq
            n_mels (float): number of mel bin
            n_class (int): number of class
        """
        super(LinearProjection, self).__init__()
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                            n_fft=n_fft,
                                                            f_min=f_min,
                                                            f_max=f_max,
                                                            n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)
        self.embedding_size = 4096
        self.linear_proj = nn.Linear(n_mels * 188, self.embedding_size) # (freq * time) to embedding dim

    def forward(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)
        x = x.squeeze(1)
        x = x.view(x.size(0), -1)
        embedding = self.linear_proj(x)
        return embedding



class Conv1dProjection(nn.Module):
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
        super(Conv1dProjection, self).__init__()

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


    def forward(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)
        x = x.squeeze(1)

        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)       
        x = self.final_pool(x)  
        x = x.squeeze(-1) 
        x = x.view(x.size(0), -1)

        return x



class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, pooling=2):
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

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out



class Conv2dProjection(nn.Module):
    def __init__(self,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=96,
                n_class=50):
        super(Conv2dProjection, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                            n_fft=n_fft,
                                                            f_min=f_min,
                                                            f_max=f_max,
                                                            n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        self.layer1 = Conv_2d(1, 64, pooling=(4,4))
        self.layer2 = Conv_2d(64, 128, pooling=(3,3))
        self.layer3 = Conv_2d(128, 128, pooling=(3,3))
        self.layer4 = Conv_2d(128, 64, pooling=(2,5))
        self.linear = nn.Linear(64, 64)
    

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
        
        return x



class TFProjection(nn.Module):
    def __init__(self,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=96):
        super(TFProjection, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                            n_fft=n_fft,
                                                            f_min=f_min,
                                                            f_max=f_max,
                                                            n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)     

        self.freq_0 = Conv_2d(1, 32, kernel_size=(48, 1))
        self.time_0 = Conv_2d(1, 32, kernel_size=(1, 64))
        self.freq_maxpool = nn.MaxPool2d((1, 95))
        self.time_maxpool = nn.MaxPool2d((49, 1))
        self.linear1 = nn.Linear(32*88, 256)


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

        return x_TF



class TFDeepProjection(nn.Module):
    def __init__(self,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=96):
        super(TFDeepProjection, self).__init__()

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

        self.linear1 = nn.Linear(64*19, 256)
        
    

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

        return x_TF



class TF2Projection(nn.Module):
    def __init__(self,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=96):
        super(TF2Projection, self).__init__()

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

        self.linear1 = nn.Linear(32*204, 256)
        

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

        return x_TF



class TripletLoss(nn.Module):
    def __init__(self, margin, options):
        """
        Args:
            margin:
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, anchor, positive, negative):
        pos_sim = nn.CosineSimilarity(dim=-1)(anchor, positive)
        neg_sim = nn.CosineSimilarity(dim=-1)(anchor, negative)
        losses = self.relu(self.margin - pos_sim + neg_sim)
        return losses.mean()


