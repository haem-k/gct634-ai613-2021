import warnings
import multiprocessing
warnings.filterwarnings(action='ignore')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torchaudio
import torch
import torch.nn as nn
from sklearn import metrics
from dataset import *
from models import *
from preprocess import *

from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from time import sleep

import IPython.display as ipd



class Runner(object):
    def __init__(self, model, lr, momentum, weight_decay, sr, tags):
        """
        Args:
            model (nn.Module): pytorch model
            lr (float): learning rate
            momentum (float): momentum
            weight_decay (float): weight_decay
            sr (float): stopping rate
            tags (list): tags with index
        """
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=5, verbose=True)
        self.learning_rate = lr
        self.stopping_rate = sr
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        self.criterion = torch.nn.BCELoss().to(self.device)
        self.tags = tags

    # Running model for train, test and validation. mode: 'train' for training, 'eval' for validation and test
    def run(self, dataloader, epoch, mode='TRAIN'):
        self.model.train() if mode == 'TRAIN' else self.model.eval()

        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f'{mode} Epoch {epoch:02}')  # progress bar
        for x, y in pbar:
            # Move mini-batch to the desired device.
            x = x.to(self.device)
            y = y.to(self.device)

            # Feed forward the model.
            prediction = self.model(x)
            
            # Compute the loss.
            loss = self.criterion(prediction, y)
            
            if mode == 'TRAIN':
                # Perform backward propagation to compute gradients.
                loss.backward()
                
                # Update the parameters.
                self.optimizer.step()
                
                # Reset the computed gradients.
                self.optimizer.zero_grad()

            batch_size = len(x)
            epoch_loss += batch_size * loss.item()

        epoch_loss = epoch_loss / len(dataloader.dataset)
        return epoch_loss


    def test(self, dataloader):
        self.model.eval()
        epoch_loss = 0
        predictions = []
        labels = []
        pbar = tqdm(dataloader, desc=f'TEST')  # progress bar

        for x, y in pbar:
            x = x.to(self.device)
            y = y.to(self.device)
            x = x.transpose(1,0) # pre-batch in audio loader (chunk, 1, waveform)

            prediction = self.model(x)
            prediction = prediction.mean(dim=0, keepdim=True) # average chunk audio
            
            loss = self.criterion(prediction, y) 
            
            batch_size = len(x)
            epoch_loss += batch_size * loss.item()
            
            predictions.extend(prediction.detach().cpu().numpy())
            labels.extend(y.detach().cpu().numpy())

        epoch_loss = epoch_loss / len(loader_test.dataset)
        roc_aucs, tag_wise_rocaucs = self.get_auc(predictions, labels)
        return roc_aucs, epoch_loss, tag_wise_rocaucs


    # Early stopping function for given validation loss, you can use this part!
    def early_stop(self, loss, epoch):
        self.scheduler.step(loss, epoch)
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        stop = self.learning_rate < self.stopping_rate
        return stop

    def get_auc(self, predictions, labels):
        roc_aucs  = metrics.roc_auc_score(labels, predictions, average='macro')
        tag_wise_predictions = np.stack(predictions).T
        tag_wise_labels = np.stack(labels).T
        tag_wise_rocaucs = {}
        for tag, logit, label in zip(self.tags, tag_wise_predictions, tag_wise_labels):
            tag_wise_rocaucs[tag] = metrics.roc_auc_score(label, logit)
        return roc_aucs, tag_wise_rocaucs



def auto_tagging(waveform, model, input_length, tags, topk):
    """
    Args:
    waveform(np.array) : no channel audio (waveform, )
    model (nn.Module): pytorch model
    input_length (int): sample_rate x duration (second) 
    tags (list): list of tags
    topk (int): tagging number
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    
    chunk_number = waveform.shape[0] // input_length
    chunk = np.zeros((chunk_number, input_length))
    
    for idx in range(chunk.shape[0]):
        chunk[idx] = waveform[idx:idx+input_length]  
    
    audio_tensor = torch.from_numpy(chunk.astype(np.float32))
    predictions = model(audio_tensor.unsqueeze(1).to(device))
    logit = predictions.mean(dim=0, keepdim=False).detach().cpu().numpy()
    
    annotation = [tags[i] for i in logit.argsort()[::-1][:topk]]
    return annotation



if __name__ == '__main__':
    # Enable and test GPU
    if not torch.cuda.is_available():
        raise SystemError('GPU device not found!')
    print(f'Found GPU at: {torch.cuda.get_device_name()}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'Torch Audio version: {torchaudio.__version__}')

    # Data Checking
    data_frame = check_data()
    print()

    # Data Preprocess
    df_train, df_valid, df_test, id_to_path = preprocess_data(data_frame)

    # Prepare data
    BATCH_SIZE = 16
    num_workers = 2
    sample_rate = 16000
    duration = 3
    input_length =  sample_rate * duration

    # Retrieve data as custom dataset
    data_path = "./data/waveform"
    tr_data = AudioDataset(data_path, input_length, df_train, id_to_path, 'TRAIN')
    va_data = AudioDataset(data_path, input_length, df_valid, id_to_path, 'VALID')
    te_data = AudioDataset(data_path, input_length, df_test, id_to_path, 'TEST')

    loader_train = DataLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, drop_last=True)
    loader_valid = DataLoader(va_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, drop_last=False)
    loader_test = DataLoader(te_data, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False) # for chunk inference

    # Training setup.
    LR = 1e-3  # learning rate
    SR = 1e-5  # stopping rate
    MOMENTUM = 0.9
    NUM_EPOCHS = 10
    WEIGHT_DECAY = 0.0  # L2 regularization weight

    # Train Baseline model
    model = Baseline()
    # model = CNN2D()
    runner = Runner(model=model, lr = LR, momentum = MOMENTUM, weight_decay = WEIGHT_DECAY, sr = SR, tags=TAGS)
    for epoch in range(NUM_EPOCHS):
        train_loss = runner.run(loader_train, epoch, 'TRAIN')
        valid_loss = runner.run(loader_valid, epoch, 'VALID')
        print("[Epoch %d/%d] [Train Loss: %.4f] [Valid Loss: %.4f]" %
            (epoch + 1, NUM_EPOCHS, train_loss, valid_loss))
        print()
        if runner.early_stop(valid_loss, epoch + 1):
            break


    # Test the trained Baseline model
    roc_aucs, epoch_loss, tag_wise_rocaucs = runner.test(loader_test)
    print(f'test_loss={epoch_loss:.5f},  roc_auc={roc_aucs:.2f}%')

    result_auc = pd.DataFrame([tag_wise_rocaucs[tag] for tag in TAGS], columns=['rocauc'], index=TAGS)
    result_auc.sort_values(by='rocauc', ascending=False).plot.bar(figsize=(18,6),rot=60) # which tag is easy and hard task


    # Sample infernece id = 224
    id = 224
    audio_sample = df_test.loc[id]
    waveform = np.load(os.path.join("./data/waveform", id_to_path[id].replace(".mp3",".npy")))
    print("annotation tag: ",list(audio_sample[audio_sample != 0].index))

    annotation = auto_tagging(waveform, runner.model, input_length, TAGS, 2)
    print("model predict tags: ",annotation)