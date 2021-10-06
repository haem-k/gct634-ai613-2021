import torch.nn as nn
import torchaudio

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
        x = self.linear(x.squeeze(-1))
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
        self.conv = None
        self.bn = None
        self.relu = None
        self.mp = None
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
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
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


'''
Metric Learning
'''
class LinearProjection(nn.Module):
    """
    Backbone model for linear proejction
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



class TripletLoss(nn.Module):
    def __init__(self, margin):
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