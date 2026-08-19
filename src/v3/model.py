import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim=128, max_length=1024):
        super().__init__()

        assert embedding_dim % 2 == 0

        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)

        frequencies = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embedding_dim))

        positional_encoding = torch.zeros(max_length, embedding_dim)

        positional_encoding[:, 0::2] = torch.sin(positions * frequencies)
        positional_encoding[:, 1::2] = torch.cos(positions * frequencies)

        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer('positional_encoding', positional_encoding) #creates a tensor that will remain unchanged

    def forward(self, embedded_tokens): #return the tokens + their added positions
        sequence_length = embedded_tokens.size(1)

        if sequence_length > self.positional_encoding.size(1):
            raise ValueError(f'Sequence length {sequence_length} exceeds ' f'maximum positional length ' f'{self.positional_encoding.size(1)}.')

        return (embedded_tokens + self.positional_encoding[:, :sequence_length])

class TransformerBugFixClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, feedforward_dim, num_classes, max_length, padding_index, dropout=.1):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_index)
        self.positional_encoding = SinusoidalPositionalEncoding(embedding_dim=embedding_dim, max_length=max_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=feedforward_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, tokens, padding_mask):
        embedded_tokens = self.embeddings(tokens)
        position_aware_tokens = self.positional_encoding(embedded_tokens)
        contextualized_tokens = self.transformer_encoder(position_aware_tokens, src_key_padding_mask=padding_mask)
        real_token_mask = (~padding_mask).int() #starts as: (batchsize, maxlen) -> (batchsize, maxlen) results in 1s/0s each diff has a vector of 1s/0s rather than the True and False
        unsqueezed = real_token_mask.unsqueeze(dim=2) #starts as: (batchsize, maxlen) -> (batchsize, maxlen, 1) for broadcasting later
        masked = contextualized_tokens * unsqueezed #results: (batchsize, maxlen, embedding_dimension) retains values for embeddings that are real tokens. pads are reduced to 0

        token_sums = masked.sum(dim=1) #sum all of the embeddings
        token_counts = unsqueezed.sum(dim=1) #sum all of the real tokens (True = 1, False = 0)

        if (token_counts == 0).any():
            raise ValueError('Cannot mean-pool a sequence containing no real tokens.')

        mean_pooling = token_sums / token_counts #results: (batchsize, embedding_dimension) results in a tensor using broadcasting
        logits = self.classifier(mean_pooling) #results: (batchsize, num_classes) pass data through the layer
        return logits
