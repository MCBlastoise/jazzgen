import music21
import fractions
from pathlib import Path
import math
import pickle

import time

def make_music_rep(time_instance_obj):
    if isinstance(time_instance_obj, music21.note.Note):
        return time_instance_obj.pitch.midi, 1
    elif isinstance(time_instance_obj, music21.chord.Chord):
        return time_instance_obj.root().midi, time_instance_obj.multisetCardinality
    elif time_instance_obj is None:
        return None, 0
    else:
        raise TypeError("Foreign music21 object, cannot make rep")

def make_rep_seq(seq):
    return [ make_music_rep(m21_obj) for m21_obj in seq ]

def get_offset_val(offset: float | fractions.Fraction) -> float:
    if isinstance(offset, float):
        return offset
    elif isinstance(offset, fractions.Fraction):
        return offset.numerator / offset.denominator
    else:
        raise TypeError("Offset was not a float or a fraction")

def preprocess_file(filename):
    s = music21.converter.parse(filename)
    assert len(s.parts) == 1

    notes, time_bounds = extract_notes(file_stream=s)
    seq = separate_sequence(notes, time_bounds)
    rep_seq = make_rep_seq(seq)
    return rep_seq

def extract_notes(file_stream):
    nc = file_stream.flatten().notes.stream()
    time_bounds = (nc.lowestOffset, nc.highestTime)
    if time_bounds[0] != 0:
        print("Start time was not 0", nc.lowestOffset)
    
    notes = []
    for n in nc:
        dur = n.duration.quarterLength
        if dur % 1 != 0 or dur < 1:
            print("Duration was not quarter-multiple valued", dur)
        _offset = n.getOffsetBySite(nc)
        offset = get_offset_val(_offset)
        note_bounds = (offset, offset + dur)
        notes.append( (n, note_bounds) )
    
    return notes, time_bounds
    
def check_overlap(slot_bounds, note_bounds):
    slot_l, slot_u = slot_bounds
    note_l, note_u = note_bounds

    note_is_subset = slot_l <= note_l and slot_u > note_u
    slot_is_subset = note_l <= slot_l and note_u > slot_u
    note_lower_overlaps = slot_l <= note_l < slot_u
    note_upper_overlaps = slot_l < note_u <= slot_u

    return note_is_subset or slot_is_subset or note_lower_overlaps or note_upper_overlaps

def separate_sequence(notes, time_bounds):
    time_start, time_end = time_bounds
    step_size = 1
    seq = [None for _ in range(0, math.ceil(time_end), step_size)]
    for t, _ in enumerate(seq):
        slot_bounds = t * step_size, (t + 1) * step_size
        for n, note_bounds in notes:
            is_overlapping = check_overlap(slot_bounds=slot_bounds, note_bounds=note_bounds)
            if is_overlapping:
                if seq[t] is not None:
                    print("Double place")
                seq[t] = n
    return seq

def get_all_files(root_folder):
    p = Path(root_folder)
    yield from (str(subpath) for subpath in p.rglob('*') if subpath.is_file())

def process_all_files(root_folder):
    all_files = get_all_files(root_folder)
    all_rep_seqs = []
    for ix, filename in enumerate(all_files):
        if ix % 1000 == 0:
            print("Processed", filename, ix)
        rep_seq = preprocess_file(filename)
        all_rep_seqs.append(rep_seq)
    return all_rep_seqs

def save_rep_seqs_to_file(filename, rep_seqs):
    with open(filename, 'wb') as f:
        pickle.dump(rep_seqs, f)

if __name__ == '__main__':
    pass

    # root_folder = 'jazznet/chords'
    # out_filename = 'rep_seqs.pkl'

    root_folder = 'jazznet/toy'
    out_filename = 'toy_rep_seqs.pkl'

    # all_rep_seqs = process_all_files(root_folder)
    # save_rep_seqs_to_file(filename=out_filename, rep_seqs=all_rep_seqs)