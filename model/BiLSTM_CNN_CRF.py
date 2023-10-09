import torch.nn as nn
import torch
from gensim.models import KeyedVectors
import numpy as np
from torchcrf import CRF

class BilstmCnnCrf(nn.Module):
    def __init__(self, 
                word2id,
                char2id, 
                num_classes, 
                word_embedding_dim=300, 
                char_embedding_dim=20,
                num_filters=30,
                hidden_dim=200, 
                num_layers=2,
                filter_size=3,
                drop_out=0.5,
                embedding_pretrained=None, 
                pretrain_vector_path=None):
        super(BilstmCnnCrf, self).__init__()
        self.WordEmbedding = WordEmbedding(word2id=word2id, 
                                         word_embedding_dim=word_embedding_dim, 
                                         embedding_pretrained=embedding_pretrained, 
                                         pretrain_vector_path=pretrain_vector_path)
        self.CharEmbedding = CharEmbedding(char2id=char2id,
                                           char_embedding_dim=char_embedding_dim,
                                            num_filters=num_filters,
                                            filter_size=filter_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_input_dim = word_embedding_dim + num_filters
        self.lstm = nn.LSTM(self.rnn_input_dim, hidden_dim // 2, num_layers, bidirectional=True, batch_first=True, dropout=drop_out)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.crf = CRF(num_classes, batch_first=True)

    def forward(self, words_to_id, chars_to_id):
        word_embedding = self.WordEmbedding(words_to_id)
        char_embedding = self.CharEmbedding(chars_to_id)
        embedding = torch.cat([word_embedding, char_embedding],2)
        self.lstm.flatten_parameters()
        out, (h, c) = self.lstm(embedding)
        out = self.fc(out)
        return out

    def compute_loss(self, words_to_id, chars_to_id, label, mask):
        out = self.forward(words_to_id, chars_to_id)
        loss = -self.crf(out, label, mask, reduction='mean')
        return loss

    def decode(self, words_to_id, chars_to_id, mask):
        out = self.forward(words_to_id, chars_to_id)
        predicted_id = self.crf.decode(out, mask)
        return predicted_id
    

class WordEmbedding(nn.Module):
    def __init__(self, 
                word2id,
                word_embedding_dim=300, 
                embedding_pretrained=None, 
                pretrain_vector_path=None):
        super(WordEmbedding, self).__init__()
        if embedding_pretrained:
            self.word_embedding = nn.Embedding(len(word2id), word_embedding_dim)
            model = KeyedVectors.load_word2vec_format(pretrain_vector_path, binary=False)
            pretrained_embeddings = model.vectors
            new_nd = np.zeros((2,300))
            pretrained_embeddings = np.r_[new_nd, pretrained_embeddings]
            self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.word_embedding.weight.requires_grad = True
        else:
            self.word_embedding = nn.Embedding(len(word2id), word_embedding_dim, padding_idx=0)
    
    def forward(self, x):
        out = self.word_embedding(x) # [batch_size, max_length, embedding_dim]
        return out 

class CharEmbedding(nn.Module):
    def __init__(self, 
            char2id,
            char_embedding_dim=20,
            num_filters=30,
            filter_size=3):
        super(CharEmbedding, self).__init__()

        self.embedding_dim = char_embedding_dim
        self.char_embedding = nn.Embedding(len(char2id), char_embedding_dim)
        self.char_embedding.weight.requires_grad = True
        self.cnn = nn.Conv3d(in_channels=1, out_channels=num_filters, kernel_size=(1, filter_size, char_embedding_dim))
 
    def forward(self, inputs):
        max_len, max_len_char = inputs.size(1), inputs.size(2)
        inputs = inputs.view(-1, max_len * max_len_char)
        input_embed = self.char_embedding(inputs)
        input_embed = input_embed.view(-1, 1, max_len, max_len_char, self.embedding_dim)

        conv_output = self.cnn(input_embed)
        pool_output = torch.squeeze(torch.max(conv_output, -2)[0])

        out = pool_output.transpose(-2, -1).contiguous() # [batch_size, max_length, num_filters]
        return out