"""
NRMS: Neural News Recommendation with Multi-head Self-Attention
===============================================================
Implements the NRMS architecture from Wu et al. (EMNLP 2019).

Architecture:
  News Encoder:
    - Input: word embedding sequence of article title (max 30 tokens)
    - Multi-head self-attention (h=4 heads, d=200 per head → 200-dim output)
    - Additive attention pooling → 200-dim news vector

  User Encoder:
    - Input: sequence of news vectors from click history (max 50 articles)
    - Multi-head self-attention over news vectors
    - Additive attention pooling → 200-dim user vector

  Scoring:
    - dot(user_vector, candidate_news_vector) → click probability

  Training:
    - Negative sampling: 1 positive + K negatives per impression
    - Loss: categorical cross-entropy (softmax over 1+K candidates)

Input/Output:
  Input:
    - History titles: (batch, max_history, max_title_len) int tokens
    - Candidate titles: (batch, n_candidates, max_title_len) int tokens
  Output:
    - Scores: (batch, n_candidates) float logits

Dimensions:
  - Vocabulary: ~50K tokens (built from training data)
  - Word embedding dim: 300
  - Self-attention heads: 4
  - Attention head dim: 50 → total output 200
  - Additive attention query dim: 200
  - News vector: 200-dim
  - User vector: 200-dim
  - Max title tokens: 30
  - Max history articles: 50
  - Negative samples per positive: 4 (during training)

Usage:
    python -m src.models.train_nrms --dataset mind --epochs 3
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Attention primitives                                               #
# ------------------------------------------------------------------ #

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention as in NRMS.
    Input:  (batch, seq_len, d_model)
    Output: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int = 200, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self._scale = math.sqrt(self.d_head)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.shape
        # Project
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,h,T,d)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self._scale  # (B,h,T,T)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)                        # (B,h,T,d)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B,T,D)
        return self.W_o(out)


class AdditiveAttentionPool(nn.Module):
    """
    Additive attention pooling — collapses a sequence to a single vector.
    q = tanh(W · h + b)  then  weight = softmax(v^T · q)
    Output: weighted sum → (batch, d_model)
    """

    def __init__(self, d_model: int = 200, d_query: int = 200):
        super().__init__()
        self.W = nn.Linear(d_model, d_query)
        self.v = nn.Linear(d_query, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, D)
        q   = torch.tanh(self.W(x))          # (B, T, d_query)
        raw = self.v(q).squeeze(-1)           # (B, T)
        if mask is not None:
            raw = raw.masked_fill(mask == 0, -1e9)
        w = F.softmax(raw, dim=-1).unsqueeze(-1)  # (B, T, 1)
        return (w * x).sum(dim=1)             # (B, D)


# ------------------------------------------------------------------ #
#  News Encoder                                                       #
# ------------------------------------------------------------------ #

class NewsEncoder(nn.Module):
    """
    Encodes a news article (title word IDs) → 200-dim vector.

    Input:  (batch, max_title_len) int64 word IDs
    Output: (batch, 200) float32 news vectors
    """

    def __init__(
        self,
        vocab_size: int,
        word_emb_dim: int  = 300,
        d_model: int       = 200,
        n_heads: int       = 4,
        max_title_len: int = 30,
        dropout: float     = 0.2,
        pretrained_emb: torch.Tensor = None,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, word_emb_dim, padding_idx=0)
        if pretrained_emb is not None:
            self.emb.weight.data.copy_(pretrained_emb)

        self.proj    = nn.Linear(word_emb_dim, d_model)
        self.attn    = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.pool    = AdditiveAttentionPool(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, title_ids: torch.Tensor) -> torch.Tensor:
        """
        title_ids : (B, max_title_len) int — 0 = padding
        returns   : (B, d_model)
        """
        mask = (title_ids != 0).float()              # (B, T)
        x    = self.dropout(self.emb(title_ids))     # (B, T, word_emb_dim)
        x    = self.proj(x)                          # (B, T, d_model)
        x    = self.norm(x + self.attn(x, mask))     # residual + self-attn
        return self.pool(x, mask)                    # (B, d_model)


# ------------------------------------------------------------------ #
#  User Encoder                                                       #
# ------------------------------------------------------------------ #

class UserEncoder(nn.Module):
    """
    Encodes a user's click history → 200-dim user vector.

    Input:  (batch, max_history, d_model) — pre-encoded news vectors
    Output: (batch, d_model)
    """

    def __init__(self, d_model: int = 200, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.pool = AdditiveAttentionPool(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, history_vecs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        history_vecs : (B, max_history, d_model)
        mask         : (B, max_history) — 1=valid, 0=padding
        returns      : (B, d_model)
        """
        x = self.norm(history_vecs + self.attn(history_vecs, mask))
        return self.pool(x, mask)  # (B, d_model)


# ------------------------------------------------------------------ #
#  Full NRMS Model                                                    #
# ------------------------------------------------------------------ #

class NRMSModel(nn.Module):
    """
    Full NRMS recommendation model.

    Input at training:
      hist_ids  : (B, max_history, max_title_len)   int64 - history article titles
      hist_mask : (B, max_history)                  bool  - valid history entries
      cand_ids  : (B, n_candidates, max_title_len)  int64 - candidate titles
                  cand_ids[:, 0, :] = positive,  cand_ids[:, 1:, :] = negatives

    Output at training:
      logits    : (B, n_candidates) — cross-entropy loss over this

    Input at inference:
      hist_ids, cand_ids  (n_candidates can be any number)
    Output at inference:
      scores    : (B, n_candidates) — higher = more relevant
    """

    def __init__(
        self,
        vocab_size: int,
        word_emb_dim: int  = 300,
        d_model: int       = 200,
        n_heads: int       = 4,
        max_title_len: int = 30,
        max_history: int   = 50,
        dropout: float     = 0.2,
        pretrained_emb: torch.Tensor = None,
    ):
        super().__init__()
        self.news_encoder = NewsEncoder(
            vocab_size, word_emb_dim, d_model, n_heads, max_title_len, dropout, pretrained_emb
        )
        self.user_encoder = UserEncoder(d_model, n_heads, dropout)
        self.d_model      = d_model
        self.max_title_len = max_title_len
        self.max_history  = max_history

    def _encode_news_batch(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of news articles, supporting an extra 'sequence' dimension.
        ids    : (..., max_title_len) — any leading dims
        returns: (..., d_model)
        """
        shape  = ids.shape[:-1]
        flat   = ids.view(-1, self.max_title_len)       # (N, T)
        vecs   = self.news_encoder(flat)                 # (N, d_model)
        return vecs.view(*shape, self.d_model)

    def forward(
        self,
        hist_ids:  torch.Tensor,
        hist_mask: torch.Tensor,
        cand_ids:  torch.Tensor,
    ) -> torch.Tensor:
        """
        hist_ids  : (B, max_history, max_title_len)
        hist_mask : (B, max_history)
        cand_ids  : (B, n_cands, max_title_len)
        returns   : (B, n_cands) logits
        """
        # Encode history → user vector
        hist_vecs  = self._encode_news_batch(hist_ids)    # (B, H, d)
        user_vec   = self.user_encoder(hist_vecs, hist_mask)  # (B, d)

        # Encode candidates
        cand_vecs  = self._encode_news_batch(cand_ids)    # (B, K, d)

        # Score = dot product
        scores = torch.bmm(cand_vecs, user_vec.unsqueeze(-1)).squeeze(-1)  # (B, K)
        return scores
