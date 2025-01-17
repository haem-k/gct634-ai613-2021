import os
from datetime import datetime

import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate
from onsets_and_frames import *
from onsets_and_frames.dataset import MAESTRO_scaled




def train(logdir, device, iterations, resume_iteration, checkpoint_interval, train_on, batch_size, sequence_length,
          model_complexity, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out,
          clip_gradient_norm, validation_length, validation_interval):

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    train_groups, validation_groups = ['train'], ['validation']

    if leave_one_out is not None:
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]

    if train_on == 'MAESTRO_small':
        dataset = MAESTRO(path='data/MAESTRO_small', groups=train_groups, sequence_length=sequence_length)
        validation_dataset = MAESTRO(path='data/MAESTRO_small', groups=validation_groups, sequence_length=sequence_length)
    elif train_on == 'MAESTRO_scaled':
        dataset = MAESTRO_scaled(groups=train_groups, sequence_length=sequence_length)
        validation_dataset = MAESTRO_scaled(groups=validation_groups, sequence_length=sequence_length)
    elif train_on == 'MAESTRO':
        dataset = MAESTRO(groups=train_groups, sequence_length=sequence_length)
        validation_dataset = MAESTRO(groups=validation_groups, sequence_length=sequence_length)
    else:
        dataset = MAPS(groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'], sequence_length=sequence_length)
        validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=validation_length)

    if train_on == 'MAESTRO_scaled':
        loader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_scaled_audio, drop_last=True)
    else:
        loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

    if resume_iteration is None:
        model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))

    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    for i, batch in zip(loop, cycle(loader)):
        if train_on == 'MAESTRO_scaled':
            predictions, losses = model.run_on_scaled_batch(batch)
        else:
            predictions, losses = model.run_on_batch(batch)

        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)

        for key, value in {'loss': loss, **losses}.items():
            writer.add_scalar(key, value.item(), global_step=i)

        if i % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                for key, value in evaluate(validation_dataset, model, dataset=train_on).items():
                    writer.add_scalar('validation/' + key.replace(' ', '_'), np.mean(value), global_step=i)
            model.train()

        if i % checkpoint_interval == 0:
            torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))


# Custom collate_fn to add zero padding to different size of data in a batch
def collate_scaled_audio(batch):
    # batch: 8 dictionary items
    # stacked_batch: 1 dictionary with stacked data

    stacked_batch = {}
    stacked_batch['audio_scaled'] = []
    stacked_batch['scaled_index'] = []
    stacked_batch['num_onsets'] = []
    stacked_batch['audio'] = []
    stacked_batch['onset'] = []
    stacked_batch['offset'] = []
    stacked_batch['frame'] = []
    stacked_batch['velocity'] = []

    # Get a data with maximum length of scaled audio
    # max_data_length = 0
    data_batch = []
    num_scaled_frames = torch.zeros((len(batch),))
    for i in range(len(batch)):
        data = batch[i]
        data_batch.append(data['audio_scaled'])
        stacked_batch['scaled_index'].append(data['scaled_index'])
        stacked_batch['num_onsets'].append(data['num_onsets'])
        stacked_batch['audio'].append(data['audio'])
        stacked_batch['onset'].append(data['onset'])
        stacked_batch['offset'].append(data['offset'])
        stacked_batch['frame'].append(data['frame'])
        stacked_batch['velocity'].append(data['velocity'])

        num_scaled_frames[i] = data['audio_scaled'].shape[0] // HOP_LENGTH
    
    # Zero pad all data
    padded_data_batch = pad_sequence(data_batch, batch_first=True)
    padded_length = padded_data_batch.shape[-1] // HOP_LENGTH

    # Stack data into one dict
    stacked_batch['padded_frames'] = torch.full_like(num_scaled_frames, padded_length) - num_scaled_frames
    stacked_batch['audio_scaled'] = padded_data_batch
    stacked_batch['scaled_index'] = torch.stack(stacked_batch['scaled_index'])
    stacked_batch['num_onsets'] = torch.stack(stacked_batch['num_onsets'])
    stacked_batch['audio'] = torch.stack(stacked_batch['audio'])
    stacked_batch['onset'] = torch.stack(stacked_batch['onset'])
    stacked_batch['offset'] = torch.stack(stacked_batch['offset'])
    stacked_batch['frame'] = torch.stack(stacked_batch['frame'])
    stacked_batch['velocity'] = torch.stack(stacked_batch['velocity'])
    
    # Return padded data
    return stacked_batch


if __name__ == '__main__':
    
    # logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S')
    # logdir = 'runs/maestro_pretrain_100000'
    logdir = 'runs/standardize_test'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 10000
    resume_iteration = None
    checkpoint_interval = 1000
    train_on = 'MAESTRO_scaled'

    batch_size = 8
    sequence_length = 200 * HOP_LENGTH
    model_complexity = 48

    learning_rate = 0.0006
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = sequence_length
    validation_interval = 500

    train(logdir, device, iterations, resume_iteration, checkpoint_interval, train_on, batch_size, sequence_length,
          model_complexity, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out,
          clip_gradient_norm, validation_length, validation_interval)
