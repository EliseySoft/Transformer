import torch

from src import Transformer

if __name__ == '__main__':
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    tmp_d = 1024
    max_seq_length = 100
    dropout = 0.1

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        tmp_d=tmp_d,
        src_vocab_size=vocab_size,
        trg_vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        dropout=dropout
    )

    src_data = torch.randint(1, vocab_size, (64, max_seq_length))
    trg_data = torch.randint(1, vocab_size, (64, max_seq_length))

    transformer_output = transformer(src_data, trg_data)
    words = torch.argmax(transformer_output, dim=2)

    print(transformer_output.shape)
