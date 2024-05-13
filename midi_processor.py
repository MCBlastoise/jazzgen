import music21
from pathlib import Path
from music_rep import MusicInstance
from jazzgen_exceptions import JazzGenException
import utils

def get_piano_stream(base_stream):
    def piano_filter(part_el):
        target_instrument = music21.instrument.Piano
        instrument = part_el.getInstrument()
        return isinstance(instrument, target_instrument)

    partitioned_instruments = music21.instrument.partitionByInstrument(base_stream)
    try:
        piano_part = next(partitioned_instruments.parts.addFilter(piano_filter))
    except StopIteration:
        raise JazzGenException("File had no piano parts")

    return piano_part

def preprocess_file(filename, interval):
    s = music21.converter.parse(filename)

    piano_stream = get_piano_stream(base_stream=s)
    music_buckets = extract_music_buckets(piano_stream=piano_stream, interval=interval)
    return music_buckets

def extract_music_buckets(piano_stream, interval):
    with_zero_duration = piano_stream.flatten().notes
    fix_graces(streamIter=with_zero_duration)
    ns = with_zero_duration.addFilter(nonzero_duration_filter).stream()

    song_bottom_offset, top_time = ns.lowestOffset, ns.highestTime
    if song_bottom_offset != 0:
        print("Start time was not 0", ns.lowestOffset)
    start_offset = song_bottom_offset - song_bottom_offset % interval # round down to the nearest interval
    num_buckets = int((top_time - start_offset) // interval)

    buckets = [None] * num_buckets
    for offset_idx in range(num_buckets):
        offset_start = offset_idx * interval + start_offset
        offset_end = offset_start + interval

        music_instance_obj = MusicInstance(interval)
        
        elems = ns.getElementsByOffset(
            offsetStart=offset_start,
            offsetEnd=offset_end,
            includeEndBoundary=False,
            includeElementsThatEndAtStart=False,
        )
        for elem in elems:
            music_instance_obj.add_m21_obj(elem)
        buckets[offset_idx] = music_instance_obj
    
    return buckets

def nonzero_duration_filter(elem):
    return elem.duration.quarterLength != 0

def fix_graces(streamIter) -> None:
    for note in streamIter:
        if note.duration.isGrace:
            new_duration = music21.duration.Duration(quarterLength=0.25)
            note.duration = new_duration

def process_all_files(root_folder, interval):
    all_files = get_all_files(root_folder)
    all_rep_seqs = []
    for _, filename in enumerate(all_files):
        # if ix % 1000 == 0:
        #     print("Processing", filename, ix)
        try:
            rep_seq = preprocess_file(filename, interval=interval)
        except JazzGenException as e:
            print(e)
            continue

        all_rep_seqs.append(rep_seq)
    return all_rep_seqs

def get_all_files(root_folder):
    p = Path(root_folder)
    yield from (str(subpath) for subpath in p.rglob('*') if subpath.is_file())

def calculate_smallest_interval(music_directory):
    p = Path(music_directory)
    smallest_interval = min( smallest_interval_in_file(music_file=subpath) for subpath in p.iterdir() )
    print("Smallest interval is", smallest_interval)
    return smallest_interval

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

if __name__ == '__main__':
    root_folder = 'tests/muse'
    out_filename = 'cached/muse_rep_seqs.pkl'

    # interval = calculate_smallest_interval(music_directory='tests/muse') # TODO: EXAMINE THIS! TRADEOFFS ARE BETWEEN USING THE MOST FINE-GRAINED INTERVAL FROM THE CORPUS OR A MORE COARSE APPROACH
    fixed_interval = 0.25 # 16th note buckets
    all_rep_seqs = process_all_files(root_folder, interval=fixed_interval)
    utils.store_pickle_data(filename=out_filename, data=all_rep_seqs)