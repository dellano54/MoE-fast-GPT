import torch
import torch.nn as nn
import torch.nn.functional as F




'''
author: Dellano Samuel Fernandez
date: 30/05/25

This is an simple implementation of GPT but it uses latest techniques to make it effecient like
MoE, Weight Tying, RMSnorm, embedding scaling

'''


'''

Embedding class
 -  i m using simple transformer embedding and learable pos encoding since,
    i ll be training it on t4 gpu with context length of 512 which is pretty small,
    so using RoPE embedding would be unnecessary

'''
class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_length: int, dropout: float = 0.1):
        super().__init__()

        self.we = nn.Embedding(vocab_size, 128)
        self.pe = nn.Embedding(max_length, embed_size)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, weight):
        x = self.we(x)
        x = self.drop(x)

        x = nn.functional.linear(x, weight)

        pos = torch.arange(x.size(1), device=x.device).repeat(x.size(0), 1)
        pos = self.pe(pos)

        x += pos

        return x
    

class LM_head(nn.Module):
    def __init__(self, embed_size: int, vocab_size: int):
        super().__init__()

        self.linear = nn.Linear(128, vocab_size, bias=False)

    def forward(self, x, weight):
        x = nn.functional.linear(x, weight.T)
        x = self.linear(x)
        return x



'''

Attention mechanism:

    - this module performs the multi head self-attention mechanism with casual mask
    - it converts the input to query, key, value and do self attention on it
    - this is based on inital GPT arch so we wont be targeting on increasing the parameters due to size constraint
    - attention formula: Attention(Q, K, V) = Softmax((Q x Kᵀ) / √dₖ) x V

'''


class AttentionMechanism(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1, causal: bool = True):
        super().__init__()

        self.qkv = nn.Linear(embed_dim, embed_dim*3, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(dropout)

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.scaling = self.head_dim ** 0.5

        self.causal = causal

        if embed_dim % n_heads != 0:
            raise ValueError("embed dim should be divisible by n_heads")

    
    def _generate_causal_mask(self, seq_len, dtype: torch.dtype, device: torch.device):
        mask = torch.ones(seq_len, seq_len, dtype=dtype, device=device)
        mask = torch.tril(mask)
        mask[mask == 0] = -1e4

        return mask
    

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        dtype = x.dtype

        q, k, v = self.qkv(x).split(self.embed_dim, dim=-1)

        q = q.contiguous().view(batch_size, self.n_heads, seq_len, self.head_dim)
        k = k.contiguous().view(batch_size, self.n_heads, seq_len, self.head_dim)
        v = v.contiguous().view(batch_size, self.n_heads, seq_len, self.head_dim)

        #q, k, v = q.float(), k.float(), v.float()

        qk = torch.matmul(q, k.transpose(-1, -2)) / self.scaling
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            qk = qk.masked_fill(mask == 0, -1e4)

        if self.causal:
            causal_mask = self._generate_causal_mask(seq_len, x.dtype, x.device)
            causal_mask = causal_mask.repeat(batch_size, self.n_heads, 1, 1)
            qk = qk + causal_mask


        qk = self.softmax(qk)
        
        v = (qk @ v)
        v = v.contiguous().view(batch_size, seq_len, self.embed_dim)


        v = self.dropout(v)
        v = self.out(v)

        return v
            
        

'''
Feed Forward network and MoE layer:
    we are going to use MoE - mixture of Experts to make the model effecient in calculations
    and reduce computations, we r gonna be using 8 experts here and each experts reduce to feed_forward//8
    to have same capacity but focusing on one task, we r going to use token wise topk MoE
    
    and we r gonna be using nn.SiLU() activation
'''


class FFN(nn.Module):
    def __init__(self, embed_dim: int, feed_forward: int, dropout: float = 0.2):
        super().__init__()

        self.FFN = nn.Sequential(
            nn.Linear(embed_dim, feed_forward, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward, embed_dim, bias=False)
        )

    def forward(self, x):
        return self.FFN(x)
    



class MoELayer(nn.Module):
    def __init__(self, embed_dim: int, feed_forward: int, num_experts: int = 4, k: int = 2, dropout: float = 0.2):
        super().__init__()

        if num_experts < k:
            raise ValueError("num experts must be greater than k")

        self.Experts = nn.ModuleList(
            [ FFN(embed_dim, feed_forward//num_experts, dropout)  for _ in range(num_experts)]
        )

        self.gating = nn.Linear(embed_dim, num_experts, bias=False)
        

        self.num_experts = num_experts
        self.k = k
        self.embed_dim = embed_dim

        self.ffn_dim = embed_dim // num_experts

    def _load_balancing_loss(self, gate_scores):
        expert_probs = gate_scores.mean(dim=0)  # (E,)
        mean = expert_probs.mean()
        var = ((expert_probs - mean) ** 2).mean()
        cv_squared = var / (mean ** 2 + 1e-10)
        return cv_squared


    def forward(self, x):
        batch_size, seq_length, _ = x.shape

        x = x.view(-1, self.embed_dim)
        N = x.shape[0]

        gating_scores = F.softmax(self.gating(x), dim=-1)
        
        topk_weights, topk_indices = torch.topk(gating_scores, self.k, dim=-1)

        output = torch.zeros(N, self.embed_dim, device=x.device, dtype=gating_scores.dtype)

        for idx in range(self.k):
            expert_ids = topk_indices[:, idx]
            expert_weights = topk_weights[:, idx]

            for expert_idx in range(self.num_experts):
                mask = (expert_ids == expert_idx)

                if not mask.any():
                    continue

                y = x[mask]
                y = self.Experts[expert_idx](y)
                y = y * expert_weights[mask].unsqueeze(1)
                scatter_idx = mask.nonzero(as_tuple=True)[0]
                output.index_add_(0, scatter_idx, y)

        loss = self._load_balancing_loss(gating_scores)

        return output.view(batch_size, seq_length, self.embed_dim), loss




'''
Decoder Layer
    - lets put it all together with layerNorm and skip connections, 
    - we r going to be using RMSNorm since its faster than normal LayerNorm and
      it is what used in LLama and it performs better

    - we gonna be using pre-norm unlike initial GPT arch cuz i think its more stable 
    but post-norm is also fine

'''

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, feed_forward: int, num_experts:int, k:int, dropout: float, causal: bool = True):
        super().__init__()

        self.norm1 = nn.RMSNorm(embed_dim)
        self.Attn = AttentionMechanism(embed_dim, n_heads, dropout, causal=True)
        
        self.norm2 = nn.RMSNorm(embed_dim)
        self.FFN = MoELayer(embed_dim, feed_forward, num_experts, k, dropout=dropout)

    
    def forward(self, x, attention_mask):
        res = x

        x = self.norm1(x)
        x = self.Attn(x, attention_mask)
        x += res

        res = x
        x = self.norm2(x)
        x, loss = self.FFN(x)
        x += res

        return x, loss
    



#finally lets put it all together

class GPT(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, n_heads: int, feed_forward: int,
                num_layers: int, max_length: int, num_experts: int, k: int, 
                dropout:float = 0.1, causal: bool=True, pad_token_id: int = 0, alpha: float = 0.3):
        super().__init__()

        self.scale_weight = nn.Parameter(torch.randn(embed_dim, 128))

        self.Embedding = Embedding(vocab_size, embed_dim, max_length, dropout)
        self.Decoder = nn.ModuleList(
            [ DecoderLayer(embed_dim, n_heads, feed_forward, num_experts, k, dropout, causal) for _ in range(num_layers)]
        )

        self.lm_head = LM_head(embed_dim, vocab_size)
        self.lm_head.weight = self.Embedding.we.weight

        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.num_layers = num_layers

        self._init_weights()


    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask = None):
        moe_loss = 0

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id)

        labels = input_ids[:, 1:].clone().long()
        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]

        x = self.Embedding(input_ids, self.scale_weight)

        for layer in self.Decoder:
            x, l = layer(x, attention_mask)
            moe_loss += l

        moe_loss = moe_loss / self.num_layers

        x = self.lm_head(x, self.scale_weight)

        ce_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)(
            x.view(-1, self.vocab_size), labels.view(-1)
        )

        loss = ce_loss + (self.alpha * moe_loss)


        return loss, ce_loss, moe_loss
    
    @torch.no_grad()
    def generate(self,
                 input_ids: torch.LongTensor,
                 max_new_tokens: int = 50,
                 temperature: float = 1.0,
                 top_k: int = None,
                 eos_token_id: int = None):
    
        batch_size = input_ids.size(0)
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            attention_mask = (generated != self.pad_token_id).long()

            x = self.Embedding(generated)
            for layer in self.Decoder:
                x, _ = layer(x, attention_mask)
            logits = self.out(x)

            next_token_logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            if top_k is not None and top_k > 0:
                topk_vals, _ = torch.topk(next_token_logits, top_k)
                min_topk = topk_vals[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_topk,
                    torch.full_like(next_token_logits, float('-inf')),
                    next_token_logits
                )

            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

  
            generated = torch.cat((generated, next_tokens), dim=1)
            if eos_token_id is not None:
                if (next_tokens == eos_token_id).all():
                    break

        return generated



