from jazzgen_model import JazzGenModel
import utils
import torch
import music_rep
import train_jazzgen
import random

def generate_music_rep_sequence(model, rep_corpus, desired_len, max_comprehension_len):
    if desired_len < 1:
        raise Exception("Desired sequence length must be at least 1")
    
    corpus_keys = list(rep_corpus)
    starting_element = random.choice(corpus_keys)
    generated_seq = [starting_element]
    
    with torch.no_grad():
        for _ in range(desired_len - 1):
            sample_start_idx = max(len(generated_seq) - max_comprehension_len, 0)
            input_sample = generated_seq[sample_start_idx:]
            
            inputs = train_jazzgen.prepare_sequence(input_sample, rep_corpus=rep_corpus)
            tag_scores = model(inputs)

            # print(torch.multinomial(input=tag_scores, num_samples=1))

            # TEMP = 2.0

            # scaled_scores = tag_scores / TEMP

            prob_scores = torch.exp(tag_scores)
            tag_idx = torch.multinomial(input=prob_scores, num_samples=1).item()

            

            # print(tag_scores, torch.exp(tag_scores))

            # tag_idx = torch.argmax(tag_scores).item()

            # tag_idx = torch.multinomial(input=tag_scores, num_samples=1).item()

            next_element = next(k for k, tag in rep_corpus.items() if tag == tag_idx)
            generated_seq.append(next_element)
    
    return generated_seq

if __name__ == '__main__':  
    # corpus_filename = 'cached/toy_rep_corpus.pkl'
    # model_filename = 'cached/toy_model.pt'
    # midi_filename = 'outputs/toy.midi'

    # corpus_filename = 'cached/muse_rep_corpus.pkl'
    # model_filename = 'cached/muse_model2.pt'
    # midi_filename = 'outputs/muse.midi'

    # corpus_filename = 'cached/toy_rep_corpus.pkl'
    # model_filename = 'cached/toy_model.pt'

    # corpus_filename = 'cached/dur_rep_corpus.pkl'
    # model_filename = 'cached/dur_model.pt'
    # midi_filename = 'outputs/muse18.midi'

    corpus_filename = 'cached/dur2_rep_corpus.pkl'
    model_filename = 'cached/dur2_model.pt'
    midi_filename = 'outputs/muse20.midi'
    seq_len = 256
    
    rep_corpus = utils.access_pickle_data(filename=corpus_filename)
    model = utils.load_model(model_filename=model_filename, rep_corpus=rep_corpus, temp=2.0)
    music_rep_seq = generate_music_rep_sequence(model=model, rep_corpus=rep_corpus, desired_len=seq_len, max_comprehension_len=40)
    m21_stream = music_rep.make_m21_stream(music_rep_seq=music_rep_seq)
    utils.stream_to_midi(stream=m21_stream, filename=midi_filename)