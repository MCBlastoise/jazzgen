import music21
import fractions
from pathlib import Path
import math
import pickle
import subprocess
from music_rep import MusicInstance

class JazzGenException(Exception):
    pass



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

def get_piano_stream(base_stream):
    def piano_filter(part_el):
        target_instrument = music21.instrument.Piano
        instrument = part_el.getInstrument()
        return isinstance(instrument, target_instrument)
        # return False
    
    # for el in base_stream.recurse().notes:
    #     print(el.getInstrument())

    # chordified = base_stream.chordify()
    # chordified.show()

    # raise Exception()

    
    # for part in base_stream.parts:
    #     print(part)

    # print("b")

    partitioned_instruments = music21.instrument.partitionByInstrument(base_stream)

    # for part in partitioned_instruments.parts:
    #     print(part) 

    # raise Exception()

    # print(partitioned_instruments.parts)

    # print(len(partitioned_instruments))
    try:
    #     # i = list(base_stream.parts.addFilter(piano_filter))[0]
    #     # print(i)
    #     # i.show('text')

    #     # printbase_stream.parts))

    #     for el in base_stream.parts:
    #         print(el.getInstrument())

        piano_part = next(partitioned_instruments.parts.addFilter(piano_filter))
        # print("Got piano part")
        # partitioned_instruments = next(base_stream.parts.addFilter(piano_filter))
    #     # piano_filter.show()
    except StopIteration:
        # print("File had no piano parts")
        raise JazzGenException("File had no piano parts")
    
    # raise Exception()

    return piano_part

def preprocess_file(filename, interval):
    s = music21.converter.parse(filename)
    # print(s.isWellFormedNotation())
    # s.show()
    # s.flatten().show()
    # s.show('text')
    # s.show('text')
    # assert len(s.parts) == 1

    # for part in s.parts:
    #     print(part.partName)

    try:
        piano_stream = get_piano_stream(base_stream=s)
        interval = 0.25 # TODO: EXAMINE THIS! TRADEOFFS ARE BETWEEN USING THE MOST FINE-GRAINED INTERVAL FROM THE CORPUS OR A MORE COARSE APPROACH
        music_buckets = extract_notes(piano_stream=piano_stream, interval=interval)
        print(music_buckets)
        # nrs = piano_stream.flatten().notesAndRests
        # for nr in nrs:
        #     if nr.duration.quarterLength == 0:
        #         continue
        #     print(nr.fullName, nr.duration.quarterLength)
        # print(min(nr.duration.quarterLength for nr in nrs if nr.duration.quarterLength != 0))
        # for nr in nrs:
        #     print(nr.duration)
        # piano_stream.show()
        # piano_stream.show('text')
        # piano_stream.show()

    except JazzGenException as e:
        print(e)



    # notes, time_bounds = extract_notes(file_stream=s)
    # seq = separate_sequence(notes, time_bounds)
    # rep_seq = make_rep_seq(seq)
    # return rep_seq

def fix_graces(streamIter) -> None:
    for note in streamIter:
        if note.duration.isGrace:
            new_duration = music21.duration.Duration(quarterLength=0.25)
            note.duration = new_duration

def extract_notes(piano_stream, interval):
    def nonzero_duration_filter(elem):
        return elem.duration.quarterLength != 0
    
    # piano_stream.show('text')

    # print("\n\n\n\n")

    with_zero_duration = piano_stream.flatten().notes
    fix_graces(streamIter=with_zero_duration)
    ns = with_zero_duration.addFilter(nonzero_duration_filter).stream()

    # ns.show()

    # raise Exception

    # ns.show('text')
    
    # ns.show()
    song_bottom_offset, top_time = ns.lowestOffset, ns.highestTime
    if song_bottom_offset != 0:
        print("Start time was not 0", ns.lowestOffset)
    start_offset = song_bottom_offset - song_bottom_offset % interval # round down to the nearest interval
    num_buckets = int((top_time - start_offset) // interval)

    buckets = [None] * num_buckets
    for offset_idx in range(num_buckets):
        offset_start = offset_idx * interval + start_offset
        offset_end = offset_start + interval
        # print(offset_start, offset_end)

        music_instance_obj = MusicInstance()
        
        elems = ns.getElementsByOffset(
            offsetStart=offset_start,
            offsetEnd=offset_end,
            includeEndBoundary=False,
            includeElementsThatEndAtStart=False,
        )
        for elem in elems:
            music_instance_obj.add_m21_obj(elem)
        buckets[offset_idx] = music_instance_obj
        
        # print(type(elems))
        # for elem in elems:
        #     print(elem, elem.duration.quarterLength)
        # print(len(elems))
        # break
    
    return buckets


    

    


# def extract_notes(file_stream):
#     nc = file_stream.flatten().notes.stream()
#     time_bounds = (nc.lowestOffset, nc.highestTime)
#     if time_bounds[0] != 0:
#         print("Start time was not 0", nc.lowestOffset)
    
#     notes = []
#     for n in nc:
#         dur = n.duration.quarterLength
#         if dur % 1 != 0 or dur < 1:
#             print("Duration was not quarter-multiple valued", dur)
#         _offset = n.getOffsetBySite(nc)
#         offset = get_offset_val(_offset)
#         note_bounds = (offset, offset + dur)
#         notes.append( (n, note_bounds) )
    
#     return notes, time_bounds
    
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

def smallest_interval_in_file(music_file):
    s = music21.converter.parse(music_file)
    try:
        piano_stream = get_piano_stream(base_stream=s)
        nrs = piano_stream.flatten().notesAndRests.stream()
        min_interval = float('inf')
        for nr in nrs:
            dur = nr.duration.quarterLength
            offset = nr.getOffsetBySite(nrs)
            if dur != 0:
                min_interval = min(dur, min_interval)
            if offset != 0:
                min_interval = min(offset, min_interval)
        return min_interval
    except JazzGenException:
        return float('inf')

def calculate_smallest_interval(music_directory):
    p = Path(music_directory)
    smallest_interval = min( smallest_interval_in_file(music_file=subpath) for subpath in p.iterdir() )
    print("Smallest interval is", smallest_interval)
    return smallest_interval

if __name__ == '__main__':
    pass

    interval = calculate_smallest_interval(music_directory='tests/muse')
    preprocess_file(filename='tests/muse/fmtm.mxl', interval=interval)











    # midi_directory = 'tests/midi'
    # mxl_target_directory = 'tests/muse'

    # process_all_files('tests/mxl')

    # convert_midi_directory(midi_directory=midi_directory, mxl_directory=mxl_target_directory)
    # show_files_in_msc(music_directory=mxl_target_directory)

    # s = music21.converter.parse('tests/midi/witp.mid')
    # s.show()

    # s = music21.converter.parse('tests/mxl/afine-1.mxl')
    # s.show()



    # root_folder = 'jazznet/chords'
    # out_filename = 'rep_seqs.pkl'

    # root_folder = 'jazznet/toy'
    # out_filename = 'toy_rep_seqs.pkl'

    # all_rep_seqs = process_all_files(root_folder)
    # save_rep_seqs_to_file(filename=out_filename, rep_seqs=all_rep_seqs)

    # filename = 'misc/MIDI_sample.mid'
    # filename = 'tests/afine-1.mid'

    # filename = 'tests/mxl/witp.mid'
    # preprocess_file(filename=filename)



    # print("Preprocessing midi\n")

    # p = Path('tests/midi')
    # for subpath in p.iterdir():
    #     print(subpath)
    #     preprocess_file(filename=subpath)
    
    # print()
    # print("Preprocessing mxl\n")

    # p = Path('tests/mxl')
    # for subpath in p.iterdir():
    #     print(subpath)
    #     preprocess_file(filename=subpath)
    