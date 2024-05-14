import music_rep
import midi_processor
import train_jazzgen
import test_jazzgen
import utils

def processor_pipeline(raw_data_dir, rep_seqs_out_filename, interval, enable):
    if not enable:
        all_rep_seqs = utils.access_pickle_data(rep_seqs_out_filename)
        return all_rep_seqs
    
    all_rep_seqs = midi_processor.process_all_files(root_folder=raw_data_dir, interval=interval)
    utils.store_pickle_data(filename=rep_seqs_out_filename, data=all_rep_seqs)
    return all_rep_seqs

def training_pipeline(rep_seqs, corpus_filename, model_filename, num_epochs, temp, enable):
    if not enable:
        rep_corpus = utils.access_pickle_data(corpus_filename)
        model = utils.load_model(model_filename=model_filename, rep_corpus=rep_corpus, temp=temp)
        return model, rep_corpus
    
    rep_corpus = train_jazzgen.make_corpus(rep_seqs=rep_seqs, corpus_cache_filename=corpus_cache_filename, use_cache=False)
    model = train_jazzgen.train_model(rep_seqs=rep_seqs, rep_corpus=rep_corpus, num_epochs=num_epochs)
    utils.save_model(model=model, filename=model_filename)
    return model, rep_corpus

def testing_pipeline(model_filename, rep_corpus, midi_filename, seq_len, temp, enable):
    if not enable:
        return
    
    model = utils.load_model(model_filename=model_filename, rep_corpus=rep_corpus, temp=temp)
    music_rep_seq = test_jazzgen.generate_music_rep_sequence(model=model, rep_corpus=rep_corpus, desired_len=seq_len, max_comprehension_len=40)
    m21_stream = music_rep.make_m21_stream(music_rep_seq=music_rep_seq)
    utils.stream_to_midi(stream=m21_stream, filename=midi_filename)

if __name__ == '__main__':
    system_name = 'laufey-course'
    
    # Pre-processor pipeline section
    raw_data_dir = 'raw_data/split'
    rep_seqs_out_filename = f'cached/{system_name}_rep_seqs.pkl'
    interval = 0.5

    # Temp is used in both
    temp = 10.0

    # Training section
    num_epochs = 3
    corpus_cache_filename = f'cached/{system_name}_rep_corpus.pkl'
    model_filename = f'cached/{system_name}_model.pt'

    # Testing section
    midi_file_directory = f'outputs/v2'
    next_file_num = utils.get_directory_size(dir_name=midi_file_directory)
    midi_filename = f'{midi_file_directory}/{next_file_num}.mid'
    seq_len = 256

    enable_preprocessor = False
    enable_trainer = False
    enable_tester = True

    rep_seqs = processor_pipeline(raw_data_dir=raw_data_dir, rep_seqs_out_filename=rep_seqs_out_filename, interval=interval, enable=enable_preprocessor)
    model, rep_corpus = training_pipeline(rep_seqs=rep_seqs, corpus_filename=corpus_cache_filename, model_filename=model_filename, num_epochs=num_epochs, temp=temp, enable=enable_trainer)
    testing_pipeline(model_filename=model_filename, rep_corpus=rep_corpus, midi_filename=midi_filename, seq_len=seq_len, temp=temp, enable=enable_tester)
