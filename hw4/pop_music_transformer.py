from miditoolkit.midi.parser import MidiFile
from model import PopMusicTransformer
from glob import glob
import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.compat.v1.disable_eager_execution()

def Generation():
    # declare model
    model = PopMusicTransformer(
        checkpoint='REMI-tempo-checkpoint',
        is_training=False)
    
    # generate from scratch
    model.generate(
        n_target_bar=16,
        temperature=1.2,
        topk=5,
        output_path='./result/from_scratch.midi',
        prompt=None)

    # generate continuation
    model.generate(
        n_target_bar=16,
        temperature=1.2,
        topk=5,
        output_path='./result/pianist8/Yiruma_River_Flows_in_You_continue.midi',
        prompt='./data/pianist8/midi/Yiruma/Yiruma_River_Flows_in_You.mid' 

    )
    
    # close model
    model.close()
    

# Generation()
# quit()

def Finetune():
    # declare model
    model = PopMusicTransformer(
        checkpoint='REMI-tempo-checkpoint',
        is_training=True)
    # prepared data
    # midi_paths = glob('./data/maestro/*.mid') # you need to revise it
    midi_paths = glob('./finetune_traindata/*.mid')
    # print(midi_paths[3])
    training_data = model.prepare_data(midi_paths=midi_paths)

    output_checkpoint_folder = 'REMI-finetune_pianist8' # your decision
    if not os.path.exists(output_checkpoint_folder):
        os.mkdir(output_checkpoint_folder)
    
    # finetune
    model.finetune(
        training_data=training_data,
        output_checkpoint_folder=output_checkpoint_folder,
        Epoch_n = 100)

    # close
    model.close()

# Finetune()
# quit()



import shutil

shutil.copyfile("./REMI-tempo-checkpoint/dictionary.pkl", "./REMI-finetune_pianist8/dictionary.pkl")




def Generation_with_finetune():
    # data_title = 'Hisaishi_Mononoke_Hime'
    # data_path = f'./data/pianist8/midi/Hisaishi/{data_title}'

    data_title = 'Yiruma_River_Flows_in_You'
    data_path = f'./data/pianist8/midi/Yiruma/{data_title}'


    # declare model
    model = PopMusicTransformer(
        checkpoint='REMI-finetune_pianist8',
        is_training=False)
    
    # generate from scratch
    # model.generate(
    #     n_target_bar=16,
    #     temperature=0.8,
    #     topk=5,
    #     output_path='./result/pianist8/finetune_from_scratch_temp0_8.midi',
    #     prompt=None)

    # model.generate(
    #     n_target_bar=16,
    #     temperature=1.2,
    #     topk=5,
    #     output_path='./result/pianist8/finetune_from_scratch_temp1_2.midi',
    #     prompt=None)
    
    # model.generate(
    #     n_target_bar=16,
    #     temperature=1.5,
    #     topk=5,
    #     output_path='./result/pianist8/finetune_from_scratch_temp1_5.midi',
    #     prompt=None)


    # generate continuation
    model.generate(
        n_target_bar=16,
        temperature=0.8,
        topk=5,
        output_path=f'./result/pianist8/{data_title}_finetune_continuation_temp0_8.midi',
        prompt=f'{data_path}.mid'  
    )

    model.generate(
        n_target_bar=16,
        temperature=1.2,
        topk=5,
        output_path=f'./result/pianist8/{data_title}_finetune_continuation_temp1_2.midi',
        prompt=f'{data_path}.mid'  
    )

    model.generate(
        n_target_bar=16,
        temperature=1.5,
        topk=5,
        output_path=f'./result/pianist8/{data_title}_finetune_continuation_temp1_5.midi',
        prompt=f'{data_path}.mid'  
    )
    
    
    # close model
    model.close()

Generation_with_finetune()


# # Convert midi to wav
# from IPython.display import Audio
# from pretty_midi import PrettyMIDI

# scratch_midi_file = './result/finetune_from_scratch.midi'
# continuation_midi_file = './result/finetune_continuation.midi'

# scratch_wav = PrettyMIDI(midi_file=scratch_midi_file)
# continuation_wav = PrettyMIDI(midi_file=continuation_midi_file)

# scratch_waveform = scratch_midi_file.fluidsynth()
# continuation_waveform = continuation_midi_file.fluidsynth()
# Audio(wave)
