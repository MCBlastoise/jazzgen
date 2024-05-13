from jazzgen_model import JazzGenModel
import utils
import torch
import music_rep
import train_jazzgen
import random

def load_model(model_filename, rep_corpus):
    vocab_size = tagset_size = len(rep_corpus)

    state_dict = torch.load(model_filename)
    model = JazzGenModel(vocab_size=vocab_size, tagset_size=tagset_size)
    model.load_state_dict(state_dict)
    model.eval()

    return model

def generate_music_rep_sequence(model, rep_corpus, desired_len):
    if desired_len < 1:
        raise Exception("Desired sequence length must be at least 1")
    
    corpus_keys = list(rep_corpus)
    starting_element = random.choice(corpus_keys)
    generated_seq = [starting_element]
    
    with torch.no_grad():
        # idx, random_datapoint = next((idx, datapoint[0]) for idx, datapoint in enumerate(training_data) if len(datapoint[0]) > 3 and idx > 3)
        # print(idx)

        for _ in range(desired_len - 1):
            inputs = train_jazzgen.prepare_sequence(generated_seq, rep_corpus=rep_corpus)
            tag_scores = model(inputs)
            tag_idx = torch.argmax(tag_scores).item()
            next_element = next(k for k, tag in rep_corpus.items() if tag == tag_idx)
            generated_seq.append(next_element)
    
    return generated_seq

if __name__ == '__main__':  
    corpus_filename = 'cached/toy_rep_corpus.pkl'
    model_filename = 'cached/toy_model.pt'
    midi_filename = 'outputs/toy.midi'
    seq_len = 5
    
    rep_corpus = utils.access_pickle_data(filename=corpus_filename)
    model = load_model(model_filename=model_filename, rep_corpus=rep_corpus)
    music_rep_seq = generate_music_rep_sequence(model=model, rep_corpus=rep_corpus, desired_len=seq_len)
    print(music_rep_seq)
    m21_stream = music_rep.make_m21_stream(music_rep_seq=music_rep_seq)
    utils.stream_to_midi(stream=m21_stream, filename=midi_filename)