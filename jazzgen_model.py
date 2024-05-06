import torch
import torch.nn as nn

class JazzGenModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(JazzGenModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.music_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
    
    def forward(self, music_in):
        embeds = self.music_embeddings(music_in)
        reshaped_embeds = embeds.view(len(music_in), 1, -1)

        _, (last_hidden, _) = self.lstm(reshaped_embeds)
        reshaped_last_hidden = torch.squeeze(last_hidden, dim=1)

        TEMPERATURE = 1.0
        tag_space = self.hidden2tag(reshaped_last_hidden)
        temp_scaled_tag_space = tag_space / TEMPERATURE

        tag_scores = nn.functional.log_softmax(temp_scaled_tag_space, dim=1)
        return tag_scores