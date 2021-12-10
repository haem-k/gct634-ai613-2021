"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

import torch
import torch.nn.functional as F
from torch import nn

from .lstm import BiLSTM
from .mel import melspectrogram
from .constants import HOP_LENGTH, SCALE


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48):
        super().__init__()

        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        self.onset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features)
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)
        combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred

    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        velocity_label = batch['velocity']

        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }

        return predictions, losses
    
    def run_on_scaled_batch(self, batch):
        scaled_index_batch = batch['scaled_index']
        audio_label = batch['audio']
        audio_scaled_label = batch['audio_scaled']
        onset_label = batch['onset']                        # (batch_size, 200, 88)
        offset_label = batch['offset']
        frame_label = batch['frame']
        velocity_label = batch['velocity']

        batch_size = onset_label.shape[0]

        mel = melspectrogram(audio_scaled_label.reshape(-1, audio_scaled_label.shape[-1])[:, :-1]).transpose(-1, -2)        
        onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)

        # TODO: Reverse standardization 
        reverted_onset_pred = torch.zeros_like(onset_label)
        reverted_offset_pred = torch.zeros_like(offset_label)
        reverted_frame_pred = torch.zeros_like(frame_label)
        reverted_velocity_pred = torch.zeros_like(velocity_label)

        num_segments = len(scaled_index_batch[0])
        num_frames_segment = audio_label.shape[1] // num_segments // HOP_LENGTH
        num_frames_scaled_segment = int(num_frames_segment * SCALE)
        num_frames_processed = audio_scaled_label.shape[1] // num_segments // HOP_LENGTH

        padded_frames = (SCALE - 1.0) * num_frames_segment
        for i in range(batch_size):
            scaled_index = scaled_index_batch[i]
            num_not_scaled = len(scaled_index[scaled_index < 0])
            num_delete_frames = int(num_not_scaled * padded_frames)
            
            cropped_onset = onset_pred[i]
            cropped_offset = offset_pred[i]
            cropped_frame = frame_pred[i]
            cropped_velocity = velocity_pred[i]

            # Crop frames if data is padded
            if num_delete_frames != 0:
                cropped_onset = onset_pred[i][:-num_delete_frames, :]
                cropped_offset = offset_pred[i][:-num_delete_frames, :]
                cropped_frame = frame_pred[i][:-num_delete_frames, :]
                cropped_velocity = velocity_pred[i][:-num_delete_frames, :]

            # For every segment, check if the segment is scaled and reshape to original length
            for j in range(num_segments):               
                first_frame = scaled_index[j]

                # Get segment from each data
                if first_frame == -1:
                    end_frame = first_frame + num_frames_segment
                else:
                    end_frame = first_frame + num_frames_scaled_segment

                onset = cropped_onset[first_frame:end_frame]
                offset = cropped_offset[first_frame:end_frame]
                frame = cropped_frame[first_frame:end_frame]
                velocity = cropped_velocity[first_frame:end_frame]
                
                # Reshape segment that is scaled
                if first_frame != -1:
                    onset = onset.transpose(0, 1).unsqueeze(0)
                    onset = F.interpolate(onset, size=num_frames_segment).squeeze().transpose(0, 1)
                    offset = offset.transpose(0, 1).unsqueeze(0)
                    offset = F.interpolate(offset, size=num_frames_segment).squeeze().transpose(0, 1)
                    frame = frame.transpose(0, 1).unsqueeze(0)
                    frame = F.interpolate(frame, size=num_frames_segment).squeeze().transpose(0, 1)
                    velocity = velocity.transpose(0, 1).unsqueeze(0)
                    velocity = F.interpolate(velocity, size=num_frames_segment).squeeze().transpose(0, 1)

                # Reshape prediction results into original length
                reverted_onset_pred[i][first_frame:first_frame+num_frames_segment] = onset
                reverted_offset_pred[i][first_frame:first_frame+num_frames_segment] = offset
                reverted_frame_pred[i][first_frame:first_frame+num_frames_segment] = frame
                reverted_velocity_pred[i][first_frame:first_frame+num_frames_segment] = velocity
            
        predictions = {
            'onset': reverted_onset_pred.reshape(*onset_label.shape),
            'offset': reverted_offset_pred.reshape(*offset_label.shape),
            'frame': reverted_frame_pred.reshape(*frame_label.shape),
            'velocity': reverted_velocity_pred.reshape(*velocity_label.shape)
        }

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }

        return predictions, losses


    def evaluate_on_scaled_batch(self, batch):
        scaled_index = batch['scaled_index']
        audio_label = batch['audio']
        audio_scaled_label = batch['audio_scaled']
        onset_label = batch['onset']                        # (batch_size, 200, 88)
        offset_label = batch['offset']
        frame_label = batch['frame']
        velocity_label = batch['velocity']

        mel = melspectrogram(audio_scaled_label.reshape(-1, audio_scaled_label.shape[-1])[:, :-1]).transpose(-1, -2)        
        onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)

        # print(f'scaled_index.shape: {scaled_index.shape}')
        # print(f'audio_label.shape: {audio_label.shape}')
        # print(f'audio_scaled_label.shape: {audio_scaled_label.shape}')
        # print(f'onset_label.shape: {onset_label.shape}')
        # print(f'offset_label.shape: {offset_label.shape}')
        # print(f'frame_label.shape: {frame_label.shape}')
        # print(f'velocity_label.shape: {velocity_label.shape}')
        # print()

        # TODO: Reverse standardization 
        reverted_onset_pred = torch.zeros_like(onset_label)
        reverted_offset_pred = torch.zeros_like(offset_label)
        reverted_frame_pred = torch.zeros_like(frame_label)
        reverted_velocity_pred = torch.zeros_like(velocity_label)

        num_segments = scaled_index.shape[0]
        num_frames_segment = audio_label.shape[0] // num_segments // HOP_LENGTH
        num_frames_scaled_segment = int(num_frames_segment * SCALE)

        padded_frames = (SCALE - 1.0) * num_frames_segment
        num_not_scaled = len(scaled_index[scaled_index < 0])
        num_delete_frames = int(num_not_scaled * padded_frames)
        
        cropped_onset = onset_pred[0]
        cropped_offset = offset_pred[0]
        cropped_frame = frame_pred[0]
        cropped_velocity = velocity_pred[0]

        # Crop frames if data is padded
        if num_delete_frames != 0:
            cropped_onset = onset_pred[0][:-num_delete_frames]
            cropped_offset = offset_pred[0][:-num_delete_frames, :]
            cropped_frame = frame_pred[0][:-num_delete_frames, :]
            cropped_velocity = velocity_pred[0][:-num_delete_frames, :]

        # For every segment, check if the segment is scaled and reshape to original length
        for j in range(num_segments):               
            first_frame = scaled_index[j]

            # Get segment from each data
            if first_frame == -1:
                end_frame = first_frame + num_frames_segment
            else:
                end_frame = first_frame + num_frames_scaled_segment

            onset = cropped_onset[first_frame:end_frame]
            offset = cropped_offset[first_frame:end_frame]
            frame = cropped_frame[first_frame:end_frame]
            velocity = cropped_velocity[first_frame:end_frame]
   
            # Reshape segment that is scaled
            if first_frame != -1:
                onset = onset.transpose(0, 1).unsqueeze(0)
                onset = F.interpolate(onset, size=num_frames_segment).squeeze().transpose(0, 1)
                offset = offset.transpose(0, 1).unsqueeze(0)
                offset = F.interpolate(offset, size=num_frames_segment).squeeze().transpose(0, 1)
                frame = frame.transpose(0, 1).unsqueeze(0)
                frame = F.interpolate(frame, size=num_frames_segment).squeeze().transpose(0, 1)
                velocity = velocity.transpose(0, 1).unsqueeze(0)
                velocity = F.interpolate(velocity, size=num_frames_segment).squeeze().transpose(0, 1)

            # Reshape prediction results into original length
            reverted_onset_pred[first_frame:first_frame+num_frames_segment] = onset
            reverted_offset_pred[first_frame:first_frame+num_frames_segment] = offset
            reverted_frame_pred[first_frame:first_frame+num_frames_segment] = frame
            reverted_velocity_pred[first_frame:first_frame+num_frames_segment] = velocity
            
        predictions = {
            'onset': reverted_onset_pred.reshape(*onset_label.shape),
            'offset': reverted_offset_pred.reshape(*offset_label.shape),
            'frame': reverted_frame_pred.reshape(*frame_label.shape),
            'velocity': reverted_velocity_pred.reshape(*velocity_label.shape)
        }

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator