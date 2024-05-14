import torch
import torch.nn as nn

class JazzGenModel(nn.Module):
    EMBEDDING_DIM = 32
    HIDDEN_DIM = 32
    
    def __init__(self, vocab_size, tagset_size, embedding_dim=None, hidden_dim=None, temp = 0.0):
        super(JazzGenModel, self).__init__()

        if embedding_dim is not None:
            self.EMBEDDING_DIM = embedding_dim
        if hidden_dim is not None:
            self.HIDDEN_DIM = hidden_dim

        self.music_embeddings = nn.Embedding(vocab_size, self.EMBEDDING_DIM)

        self.lstm = nn.LSTM(self.EMBEDDING_DIM, self.HIDDEN_DIM, batch_first=False)
        self.hidden2tag = nn.Linear(self.HIDDEN_DIM, tagset_size)

        self.temp = temp
    
    def forward(self, music_in):
        embeds = self.music_embeddings(music_in)
        reshaped_embeds = embeds.view(len(music_in), 1, -1)

        _, (last_hidden, _) = self.lstm(reshaped_embeds)
        reshaped_last_hidden = torch.squeeze(last_hidden, dim=1)
        
        tag_space = self.hidden2tag(reshaped_last_hidden)
        temp_scaled_tag_space = tag_space / self.temp if self.temp != 0 else tag_space

        tag_scores = nn.functional.log_softmax(temp_scaled_tag_space, dim=1)
        # tag_scores = nn.functional.softmax(temp_scaled_tag_space, dim=1)
        return tag_scores