import torch
import numpy as np
import sentencepiece as spm

class Tokenizer:
    def __init__(self, model_file: str, max_length: int = 512):
        self.tokenizer   = spm.SentencePieceProcessor(model_file=model_file)
        self.max_length  = max_length

        self.pad_token_id = self.tokenizer.pad_id()
        self.eos_token_id = self.tokenizer.eos_id()
        self.bos_token_id = self.tokenizer.bos_id()
        self.unk_token_id = self.tokenizer.unk_id()

        self.pad_token = self.tokenizer.decode(self.pad_token_id)
        self.eos_token = self.tokenizer.decode(self.eos_token_id)
        self.bos_token = self.tokenizer.decode(self.bos_token_id)
        self.unk_token = self.tokenizer.decode(self.unk_token_id)

    def __len__(self):
      return len(self.tokenizer)
        
    def __call__(self, batch: list[str]) -> dict[str, torch.Tensor]:
        # Batched encoding using SentencePiece's built-in parallelism
        encoded = self.tokenizer.Encode(
            batch,
            out_type=int,
            add_bos=True,
            add_eos=True
        )

        

        # Truncate & pad efficiently
        input_ids      = np.full((len(encoded), self.max_length), self.pad_token_id, dtype=np.int32)
        attention_mask = np.zeros((len(encoded), self.max_length), dtype=np.int32)

        for i, seq in enumerate(encoded):
            if len(seq) > self.max_length:
                seq = seq[:self.max_length - 1] + [self.eos_token_id]
            seq_len = len(seq)
            input_ids[i, :seq_len] = seq
            attention_mask[i, :seq_len] = 1

        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention_mask),
        }


    def decode(self, batch):
      batch = batch.tolist()
      return self.tokenizer.Detokenize(batch)