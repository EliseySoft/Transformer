import torch
import torch.nn as nn

from src.transformer_blocks import Encoder, Decoder


class Transformer(nn.Module):
    def __init__(
            self,
            num_layers: int,
            d_model: int,
            num_heads: int,
            tmp_d: int,
            src_vocab_size: int,
            trg_vocab_size: int,
            max_seq_length: int,
            dropout: float,
            pad_idx: int = 0
    ):
        super().__init__()

        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            tmp_d=tmp_d,
            src_vocab_size=src_vocab_size,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            tmp_d=tmp_d,
            trg_vocab_size=trg_vocab_size,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        self.pad_idx = pad_idx

    def generate_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def generate_trg_mask(self, trg):
        trg_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)
        seq_length = trg.shape[1]
        triangular_mask = torch.tril(torch.ones(seq_length, seq_length), diagonal=0).unsqueeze(0).bool()
        trg_mask = trg_mask & triangular_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.generate_src_mask(src)
        trg_mask = self.generate_trg_mask(trg)

        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(src, encoder_output, src_mask, trg_mask)

        return decoder_output
