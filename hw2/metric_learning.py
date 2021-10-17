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
import utils
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

torch.multiprocessing.set_sharing_strategy('file_system')


class Metric_Runner(object):
    def __init__(self, model, options):
        """
        Args:
            model (nn.Module): pytorch model
            lr (float): learning rate
            momentum (float): momentum
            weight_decay (float): weight_decay
            sr (float): stopping rate
        """
        if options.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=options.momentum, nesterov=True, weight_decay=options.weight_decay)
        elif options.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=5, verbose=True)
        self.learning_rate = options.lr
        self.stopping_rate = options.sr
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        self.criterion = TripletLoss(margin=0.4).to(self.device)

    # Running model for train, test and validation. mode: 'train' for training, 'eval' for validation and test
    def run(self, dataloader, epoch, mode='TRAIN'):
        self.model.train() if mode == 'TRAIN' else self.model.eval()

        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f'{mode} Epoch {epoch:02}')  # progress bar
        for item in pbar:
            # Move mini-batch to the desired device.
            anc, pos, neg = item
            anc_emb = self.model(anc.to(self.device))
            pos_emb = self.model(pos.to(self.device))
            neg_emb = self.model(neg.to(self.device))
            
            # Compute the loss.
            loss = self.criterion(anc_emb, pos_emb, neg_emb)
            if mode == 'TRAIN':
                # Perform backward propagation to compute gradients.
                loss.backward()
                # Update the parameters.
                self.optimizer.step()
                # Reset the computed gradients.
                self.optimizer.zero_grad()

            batch_size = anc_emb.shape[0]
            epoch_loss += batch_size * loss.item()
        epoch_loss = epoch_loss / len(dataloader.dataset)
        return epoch_loss

    def test(self, dataloader):
        self.model.eval()
        epoch_loss = 0
        embeddings, labels = [], []
        pbar = tqdm(dataloader, desc=f'TEST')  # progress bar
        for waveform, label in pbar:
            waveform = waveform.transpose(1,0)
            with torch.no_grad():
                embedding = self.model(waveform.to(self.device))
            embeddings.append(embedding.mean(0,True).detach().cpu())
            labels.append(label)
        embeddings = torch.stack(embeddings).squeeze(1)     # [1677, 4096]
        labels = torch.stack(labels).squeeze(1)             # [1677, 50]

        # calculate cosine similarity (if you use different distance metric, than you need to change this part)
        embedding_norm = embeddings / embeddings.norm(dim=-1, keepdim=True)
        sim_matrix = embedding_norm @ embedding_norm.T
        sim_matrix = sim_matrix.detach().cpu()
        labels = labels.detach().cpu().numpy()
        multilabel_recall = {
            "R@1" : self.multilabel_recall(sim_matrix, labels, top_k=1),
            "R@2" : self.multilabel_recall(sim_matrix, labels, top_k=2),
            "R@4" : self.multilabel_recall(sim_matrix, labels, top_k=4),
            "R@8" : self.multilabel_recall(sim_matrix, labels, top_k=8),
        }
        return multilabel_recall

    def multilabel_recall(self, sim_matrix, binary_labels, top_k):
        # =======================
        # TODO
        # sim_matrix:       (1677, 1677)
        # binary_labels:    (1677, 50)

        recall = 0.0
        num_test_samples = int(sim_matrix.shape[0])
        num_labels = int(binary_labels.shape[1])

        # Get top K samples that are similar to each sample
        _, indices = sim_matrix.topk(top_k+1)                # [1677, k] [1677, k]
        
        # For each test sample, compute correct answer ratio and average them
        for i in range(num_test_samples):
            # Get GT labels for i-th test sample
            gt_labels = binary_labels[i]                        # [50]
            
            # Get indices for top-K similar samples
            top_k_indices = indices[i]                          # [k]
               
            # For all top-K samples, add all number of labels that are contained in the samples 
            correct = np.zeros((num_labels))
            for k in range(top_k+1):
                # Do not consider similarity with oneself
                if k==0:
                    continue

                k_index = top_k_indices[k]
                k_gt_labels = binary_labels[k_index]
                
                # Count the identical labels
                compare = list(gt_labels == k_gt_labels)
                for j in range(num_labels):
                    if compare[j]:
                        correct[j] = 1

            ratio = np.sum(correct) / gt_labels.shape[0]
            if ratio > 1:
                ratio = 1
            recall += ratio

        recall /= num_test_samples
        return recall
        # =======================

    # Early stopping function for given validation loss
    def early_stop(self, loss, epoch):
        self.scheduler.step(loss, epoch)
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        stop = self.learning_rate < self.stopping_rate
        return stop





if __name__ == '__main__':
    # Data Checking
    data_frame = check_data()
    print()

    # Data Preprocess
    df_train, df_valid, df_test, id_to_path = preprocess_data(data_frame)
    print()

    # Get user options
    options = utils.train_metric()
    print(f"Received options:\n{options}\n")

    # Prepare data
    BATCH_SIZE = options.batch_size
    num_workers = options.num_workers
    sample_rate = options.sample_rate
    duration = options.duration
    input_length =  sample_rate * duration

    # Retrieve data as custom dataset
    data_path = "./data/waveform"
    tr_data = TripletDataset(data_path, id_to_path, input_length, df_train, TAGS, "TRAIN")
    va_data = TripletDataset(data_path, id_to_path, input_length, df_valid, TAGS, "VALID")
    te_data = TripletDataset(data_path, id_to_path, input_length, df_test, TAGS, "TEST")

    loader_train = DataLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, drop_last=True)
    loader_valid = DataLoader(va_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, drop_last=False)
    loader_test = DataLoader(te_data, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False) # for chunk inference

    # Training setup.
    NUM_EPOCHS = options.num_epochs
    # LR = 1e-3  # learning rate
    # SR = 1e-5  # stopping rate
    # MOMENTUM = 0.9
    # WEIGHT_DECAY = 0.0  # L2 regularization weight        -> Replaced with argparser
    
    model = LinearProjection() 
    runner = Metric_Runner(model=model, options=options)
    for epoch in range(NUM_EPOCHS):
        train_loss = runner.run(loader_train, epoch, 'TRAIN')
        valid_loss = runner.run(loader_valid, epoch, 'VALID')
        print("[Epoch %d/%d] [Train Loss: %.4f] [Valid Loss: %.4f]" %
                (epoch + 1, NUM_EPOCHS, train_loss, valid_loss))
        if runner.early_stop(valid_loss, epoch + 1):
            break

    # TODO: multilabel_recall
    multilabel_recall = runner.test(loader_test)
    print(multilabel_recall)