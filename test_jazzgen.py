from jazzgen_model import JazzGenModel
import utils
import torch

def load_model(model_filename, corpus_filename):
    rep_corpus = utils.access_pickle_data(filename=corpus_filename)
    vocab_size = tagset_size = len(rep_corpus)

    state_dict = torch.load(model_filename)
    model = JazzGenModel(vocab_size=vocab_size, tagset_size=tagset_size)
    model.load_state_dict(state_dict)
    model.eval()

    return model

if __name__ == '__main__':  
    corpus_filename = 'cached/toy_rep_corpus.pkl'
    model_filename = 'cached/toy_model.pt'
    
    model = load_model(model_filename=model_filename, corpus_filename=corpus_filename)
    print(model)