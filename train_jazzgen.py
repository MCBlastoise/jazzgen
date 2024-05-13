import torch
import pickle
from pathlib import Path
from jazzgen_model import JazzGenModel
# import torch.nn
# import torch.optim as optim
import utils

def make_corpus(rep_seqs, corpus_cache_filename=None, use_cache=True):
    if use_cache and corpus_cache_filename is not None and Path(corpus_cache_filename).exists():
        return utils.access_pickle_data(corpus_cache_filename)
    
    rep_corpus = {}
    for seq in rep_seqs:
        for rep in seq:
            if rep not in rep_corpus:
                rep_corpus[rep] = len(rep_corpus)
    
    if corpus_cache_filename is not None:
        utils.store_pickle_data(filename=corpus_cache_filename, data=rep_corpus)
    
    return rep_corpus

def make_training_pairs(rep_seq):
    MIN_SEQ_SIZE = 2
    MAX_SEQ_SIZE = 10
    if len(rep_seq) < MIN_SEQ_SIZE:
        print("Sequence too short", rep_seq)
    else:
        for ix, _ in enumerate(rep_seq[:-1]):
            start_seq_idx = max(ix + 1 - MAX_SEQ_SIZE, 0)
            end_seq_idx = ix + 1
            music_prefix = rep_seq[ start_seq_idx : end_seq_idx ]

            next_sound = rep_seq[ix + 1 : ix + 2]

            pair = music_prefix, next_sound
            yield pair

def make_training_data(rep_seqs):
    for rep_seq in rep_seqs:
        yield from make_training_pairs(rep_seq)

def prepare_sequence(seq, rep_corpus):
    idxs = [rep_corpus[rep] for rep in seq]
    return torch.tensor(idxs, dtype=torch.long)

def train_model(rep_seqs, rep_corpus: dict[tuple[int, int], int]):
    # These will usually be more like 32 or 64 dimensional.
    # We will keep them small, so we can see how the weights change as we train.
    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6

    # print("corpus", rep_corpus)
        
    # model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    model = JazzGenModel(
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        vocab_size=len(rep_corpus),
        tagset_size=len(rep_corpus)
    )
    loss_function = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    NUM_EPOCHS = 5
    for epoch in range(NUM_EPOCHS):  # again, normally you would NOT do 300 epochs, it is toy data
        print(f"Starting epoch {epoch + 1}")
        
        training_data = list(make_training_data(rep_seqs=rep_seqs))
        for music_prefix, next_sound in training_data:

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of indices.

            music_in = prepare_sequence(seq=music_prefix, rep_corpus=rep_corpus)
            target_out = prepare_sequence(seq=next_sound, rep_corpus=rep_corpus)

            # Step 3. Run our forward pass.
            tag_scores = model(music_in)
            reshaped_tag_scores = tag_scores.view(1, -1)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(reshaped_tag_scores, target_out)

            loss.backward()
            optimizer.step()

    return model

def save_model(model, filename):
    state_dict = model.state_dict()
    torch.save(state_dict, filename)

if __name__ == '__main__':
    rep_seqs_filename = 'cached/muse_rep_seqs.pkl'
    corpus_cache_filename = 'cached/muse_rep_corpus.pkl'
    model_filename = 'cached/muse_model.pt'

    rep_seqs = utils.access_pickle_data(filename=rep_seqs_filename)
    rep_corpus = make_corpus(rep_seqs=rep_seqs, corpus_cache_filename=corpus_cache_filename, use_cache=False)
    model = train_model(rep_seqs=rep_seqs, rep_corpus=rep_corpus)
    save_model(model=model, filename=model_filename)