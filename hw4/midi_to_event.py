import utils
import modules
import model
import chord_recognition
import miditoolkit
from collections import Counter
import pickle
import glob
import os

# Read example midi file
# file_name = 'sample_data/sample_data.midi'
data_path = 'data/maestro'
file_names = sorted(glob.glob(os.path.join(data_path, '*.midi')))

for midi in file_names:
    midi_obj = miditoolkit.midi.parser.MidiFile(midi)

    # print('-MIDI META DATA-')
    # print(midi_obj)
    # print()
    # print('-Instrument information(GM number)-')
    # print(*midi_obj.instruments, sep='\n')

    # # Show 10 note events of instrument 0
    # print(*midi_obj.instruments[0].notes[:10], sep='\n')

    # Convert to REMI events
    # Convert MIDI scores into text-like discrete tokens
    note_items, tempo_items = utils.read_items(midi)
    # print(*note_items[:10], sep='\n')
    # print()
    # print(*tempo_items[:10], sep='\n')

    # Quantize items
    # ticks: division of time in MIDI files
    note_items = utils.quantize_items(note_items, ticks=64)
    # print(*note_items[:10], sep='\n')
    # print()

    # Extract chord
    chord_items = utils.extract_chords(note_items)
    # print(*chord_items[:10], sep='\n')

    # Group items
    items = chord_items + tempo_items + note_items
    max_time = note_items[-1].end
    groups = utils.group_items(items, max_time)
    # for g in groups:
    #     print(*g, sep='\n')
    #     print()

    events = utils.item2event(groups)
    print(*events[:30], sep='\n')

    # Save events as a file
    pickle.dump(events, open('events.pkl', 'wb'))

    # Convert events to MIDI
    events = pickle.load(open('events.pkl', 'rb'))
    midi_path = midi[:-4] + '_converted.midi'
    utils.write_midi_events(None, None, midi_path, prompt_path=None, events=events)

