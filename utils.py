import pickle
from pathlib import Path
import subprocess
import music21
from jazzgen_model import JazzGenModel
import torch

def access_pickle_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def store_pickle_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def stream_to_midi(stream, filename):
    fp = stream.write('midi', fp=filename)
    return fp


def convert_midi_directory(midi_directory, mxl_directory):
    p = Path(midi_directory)
    for subpath in p.iterdir():
        convert_file_to_mxl(midi_filename=subpath, mxl_directory=mxl_directory)

def convert_file_to_mxl(midi_filename, mxl_directory):
    p = Path(midi_filename)
    mxl_filename = f'{p.stem}.mid'
    mxl_pathname = str(Path(mxl_directory) / mxl_filename)

    conversion_tokens = ['mscore', '-o', mxl_pathname, midi_filename]
    subprocess.run(conversion_tokens, stderr=subprocess.DEVNULL)

def show_files_in_msc(music_directory):
    p = Path(music_directory)
    for subpath in p.iterdir():
        s = music21.converter.parse(subpath)
        s.show()


def load_model(model_filename, rep_corpus, temp=0.0):
    vocab_size = tagset_size = len(rep_corpus)

    state_dict = torch.load(model_filename)
    model = JazzGenModel(vocab_size=vocab_size, tagset_size=tagset_size, temp=temp)
    model.load_state_dict(state_dict)
    model.eval()

    return model

def save_model(model, filename):
    state_dict = model.state_dict()
    torch.save(state_dict, filename)


def get_directory_size(dir_name):
    p = Path(dir_name)
    return len(list(p.iterdir()))