import pickle
from pathlib import Path
import subprocess
import music21

def access_pickle_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def store_pickle_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def stream_to_midi(stream, filename):
    raise NotImplementedError

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