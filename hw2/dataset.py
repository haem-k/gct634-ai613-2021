from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random

class AudioDataset(Dataset):
    def __init__(self, paths, input_length, binary, id_to_path, split):
        """
        Args:
            paths (str): path to load dataset from
            input_length (int): sample_rate x duration (second) 
            binary (Pandas.DataFrame): binary matrix for audio (index: track_id, columns: tag binary)
            id_to_path (Dict): track id to audio path
            split (str): one of [TRAIN, VALID, TEST]
        """
        self.paths = paths
        self.input_length = input_length
        self.binary = binary
        self.id_to_path = id_to_path
        self.split = split

    def __getitem__(self, index):
        item = self.binary.iloc[index]
        waveform = self.item_to_waveform(item)
        return waveform.astype(np.float32), item.values.astype(np.float32)

    def item_to_waveform(self, item):
        id = item.name
        path = os.path.join(self.paths, self.id_to_path[id].replace(".mp3", ".npy")) # pre-extract waveform, for fast loader
        waveform = np.load(path) 
        if self.split in ['TRAIN','VALID']:
            random_idx = np.random.randint(low=0, high=int(waveform.shape[0] - self.input_length))
            waveform = waveform[random_idx:random_idx+self.input_length] # extract input
            audio = np.expand_dims(waveform, axis = 0)# 1 x samples
        elif self.split == 'TEST':
            chunk_number = waveform.shape[0] // self.input_length
            chunk = np.zeros((chunk_number, self.input_length))
            for idx in range(chunk.shape[0]):
                chunk[idx] = waveform[idx:idx+self.input_length]
            audio = chunk
        return audio
            
    def __len__(self):
        return len(self.binary)



class TripletDataset(Dataset):
    def __init__(self, paths, id_to_path, input_length, binary, tags, split):
        """
        Args:
            paths (str): path to load dataset from
            id_to_path (Dict): track id to audio path
            input_length (int): sample_rate x duration (second) 
            binary (Pandas.DataFrame): binary matrix for audio (index: track_id, columns: tag binary)
            tags(list) : list of tag
            split (str): one of [TRAIN, VALID, TEST]
        """
        self.paths = paths
        self.id_to_path = id_to_path
        self.input_length = input_length
        self.tags = tags
        self.binary = binary
        self.split = split

    def __getitem__(self, index):
        if self.split in ["TRAIN","VALID"]:
            tag = random.choice(self.tags)
            tag_binary = self.binary[tag]
            positive_tracks = tag_binary[tag_binary != 0]
            negative_tracks = tag_binary[tag_binary == 0]
            anc_id, pos_id = positive_tracks.sample(2).index
            neg_id = negative_tracks.sample(1).index[0]

            anc_waveform = self.item_to_waveform(anc_id)
            pos_waveform = self.item_to_waveform(pos_id)
            neg_waveform = self.item_to_waveform(neg_id)
            return anc_waveform, pos_waveform, neg_waveform

        elif self.split == "TEST":
            item = self.binary.iloc[index]
            id = item.name
            path = os.path.join(self.paths, self.id_to_path[id].replace(".mp3",".npy"))
            waveform = np.load(path)
            chunk_number = waveform.shape[0] // self.input_length
            chunk = np.zeros((chunk_number, self.input_length))
            for idx in range(chunk.shape[0]):
                chunk[idx] = waveform[idx:idx+self.input_length]
            audio = chunk.astype(np.float32)
            label = item.values.astype(np.float32)
            return audio, label

    def item_to_waveform(self, id):
        path = os.path.join(self.paths, self.id_to_path[id].replace(".mp3", ".npy")) # pre-extract waveform, for fast loader
        waveform = np.load(path) 
        random_idx = np.random.randint(low=0, high=int(waveform.shape[0] - self.input_length))
        waveform = waveform[random_idx:random_idx+self.input_length] # extract input
        audio = np.expand_dims(waveform, axis = 0)# 1 x samples
        return audio.astype(np.float32)

    def __len__(self):
        if self.split in ["TRAIN","VALID"]:
            return len(self.binary) # it is tag wise sampling, so you can increase dataset length
        elif self.split == "TEST":
            return len(self.binary)