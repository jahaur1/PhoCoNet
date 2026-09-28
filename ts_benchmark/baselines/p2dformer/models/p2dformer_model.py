import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbedding(nn.Module):
    """
    Endogenous patch embedding.
    Input:  x [B, T, C]
    Output: h [B, C, P, D]
    """
    def __init__(self, seq_len, patch_len, d_model, d_ff, enc_in):
        super(PatchEmbedding, self).__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.patch_num = math.ceil(seq_len / patch_len)
        self.pad_num = self.patch_num * patch_len - seq_len

        self.proj = nn.Sequential(
            nn.LayerNorm(patch_len),
            nn.Linear(patch_len, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model)
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, enc_in, self.patch_num, d_model) * 0.02
        )

    def forward(self, x):
        # x: [B, T, C] -> [B, C, T]
        x = x.permute(0, 2, 1)

        # padding on temporal dimension
        x = F.pad(x, (0, self.pad_num))

        # [B, C, P, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)

        # [B, C, P, D]
        x = self.proj(x)
        x = x + self.pos_embedding

        return x


class ExogenousEmbedding(nn.Module):
    """
    Future exogenous covariate embedding.
    Input:  x_ex [B, pred_len, M]
    Output: e_ex [B, M, D]
    """
    def __init__(self, pred_len, exog_dim, d_model, d_ff):
        super(ExogenousEmbedding, self).__init__()

        self.exog_dim = exog_dim
        self.pred_len = pred_len

        self.proj = nn.Sequential(
            nn.LayerNorm(pred_len),
            nn.Linear(pred_len, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model)
        )

        self.variate_embedding = nn.Parameter(
            torch.randn(1, exog_dim, d_model) * 0.02
        )

    def forward(self, x_ex):
        # x_ex: [B, pred_len, M] -> [B, M, pred_len]
        x_ex = x_ex.permute(0, 2, 1)

        # [B, M, D]
        e_ex = self.proj(x_ex)
        e_ex = e_ex + self.variate_embedding

        return e_ex


class MSUnit(nn.Module):
    """
    Multi-scale unit using depthwise separable convolution.
    kernels = {1, 3, 5, 7}
    """
    def __init__(self, d_model, dropout=0.1):
        super(MSUnit, self).__init__()

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=1, padding=0, groups=d_model),
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model),
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model),
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.GELU()
            )
        ])

        self.fusion = nn.Sequential(
            nn.Conv1d(d_model * 4, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.gate = nn.Sequential(
            nn.Conv1d(d_model * 4, d_model, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, L, D]
        """
        residual = x

        # [B, D, L]
        x = x.transpose(1, 2)

        outs = [branch(x) for branch in self.branches]
        multi = torch.cat(outs, dim=1)

        y = self.fusion(multi) * self.gate(multi)

        # [B, L, D]
        y = y.transpose(1, 2)

        return y + residual


class P2D_FFN(nn.Module):
    """
    Period-aware 2D feed-forward network.

    It detects dominant periods by FFT, reshapes the sequence into
    [inter-period, intra-period] 2D structure, applies 2D convolution,
    then folds back to 1D sequence.
    """
    def __init__(self, d_model, d_ff, top_k=3, dropout=0.1):
        super(P2D_FFN, self).__init__()

        self.top_k = top_k
        self.d_model = d_model

        self.conv2d = nn.Sequential(
            nn.Conv2d(d_model, d_ff, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_ff, d_ff, kernel_size=3, padding=1, groups=d_ff),
            nn.GELU(),
            nn.Conv2d(d_ff, d_model, kernel_size=1)
        )

        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )

        self.alpha = nn.Parameter(torch.ones(1) * 0.1)

    def _get_periods(self, x):
        """
        x: [B, L, D]
        Return dominant periods and weights.
        """
        B, L, D = x.shape

        # FFT along token dimension
        xf = torch.fft.rfft(x, dim=1)

        # amplitude: [freq]
        amp = xf.abs().mean(dim=(0, 2))

        # remove zero frequency
        amp[0] = 0

        k = min(self.top_k, amp.shape[0] - 1)
        _, top_indices = torch.topk(amp, k=k)

        # period = L / frequency index
        periods = []
        weights = []

        for idx in top_indices:
            idx = idx.item()
            period = max(1, L // idx)
            periods.append(period)
            weights.append(amp[idx])

        weights = torch.stack(weights)
        weights = torch.softmax(weights, dim=0)

        return periods, weights

    def forward(self, x):
        """
        x: [B, L, D]
        """
        B, L, D = x.shape
        periods, weights = self._get_periods(x)

        outputs = []

        for period in periods:
            if L % period != 0:
                pad_len = period - (L % period)
                x_pad = F.pad(x, (0, 0, 0, pad_len))
            else:
                pad_len = 0
                x_pad = x

            L_pad = x_pad.shape[1]
            num_period = L_pad // period

            # [B, L_pad, D] -> [B, D, num_period, period]
            x_2d = x_pad.reshape(B, num_period, period, D)
            x_2d = x_2d.permute(0, 3, 1, 2).contiguous()

            y_2d = self.conv2d(x_2d)

            # [B, D, num_period, period] -> [B, L_pad, D]
            y = y_2d.permute(0, 2, 3, 1).reshape(B, L_pad, D)

            if pad_len > 0:
                y = y[:, :L, :]

            outputs.append(y)

        y = 0
        for i, out in enumerate(outputs):
            y = y + weights[i] * out

        y = self.out(y)

        return x + self.alpha * y


class IntersectAttention(nn.Module):
    """
    Global-token-mediated exogenous fusion.

    Query: global token
    Key/Value: future exogenous tokens
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(IntersectAttention, self).__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, global_token, exog_token):
        """
        global_token: [B, C, 1, D]
        exog_token:   [B, M, D]
        """
        B, C, _, D = global_token.shape

        # Merge target/endogenous variate dimension
        q = global_token.reshape(B * C, 1, D)

        # Repeat exogenous tokens for each endogenous variable
        kv = exog_token.unsqueeze(1).repeat(1, C, 1, 1)
        kv = kv.reshape(B * C, exog_token.shape[1], D)

        q_norm = self.norm(q)

        out, _ = self.attn(q_norm, kv, kv)

        out = q + self.dropout(out)

        # [B, C, 1, D]
        out = out.reshape(B, C, 1, D)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, top_k=3, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.msunit = MSUnit(d_model, dropout)
        self.p2d_ffn = P2D_FFN(d_model, d_ff, top_k, dropout)
        self.intersect_attn = IntersectAttention(d_model, n_heads, dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, h, exog_token=None):
        """
        h: [B, C, P+1, D], first token is global token
        exog_token: [B, M, D]
        """
        B, C, L, D = h.shape

        # Variate-wise independent patch self-attention
        h_ = h.reshape(B * C, L, D)
        h_norm = self.norm1(h_)

        attn_out, _ = self.self_attn(h_norm, h_norm, h_norm)
        h_ = h_ + self.dropout(attn_out)

        h = h_.reshape(B, C, L, D)

        # Extract global token
        g = h[:, :, :1, :]          # [B, C, 1, D]
        patches = h[:, :, 1:, :]   # [B, C, P, D]

        # Enhance global token by MSUnit + P2D-FFN
        g_seq = g.squeeze(2)       # [B, C, D]
        g_seq = self.msunit(self.norm2(g_seq))
        g_seq = self.p2d_ffn(self.norm3(g_seq))
        g = g_seq.unsqueeze(2)

        # Intersect-Attention with future exogenous variables
        if exog_token is not None:
            g = self.intersect_attn(g, exog_token)

        # Broadcast global semantics back to patches
        patches = patches + g

        h = torch.cat([g, patches], dim=2)

        # FFN
        h_ = h.reshape(B * C, L, D)
        h_ = h_ + self.ffn(self.norm4(h_))
        h = h_.reshape(B, C, L, D)

        return h


class ForecastHead(nn.Module):
    """
    Decode patch tokens to future prediction.
    """
    def __init__(self, patch_num, d_model, d_ff, pred_len, c_out):
        super(ForecastHead, self).__init__()

        self.pred_len = pred_len
        self.c_out = c_out

        self.head = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear((patch_num + 1) * d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, pred_len)
        )

    def forward(self, h):
        """
        h: [B, C, P+1, D]
        output: [B, pred_len, C]
        """
        y = self.head(h)           # [B, C, pred_len]
        y = y.permute(0, 2, 1)     # [B, pred_len, C]

        if self.c_out is not None:
            y = y[:, :, :self.c_out]

        return y


class Model(nn.Module):
    """
    P2D-Former: Period-aware 2D Transformer for time series forecasting.

    Main idea:
    1. historical endogenous variables -> patch tokens
    2. add global token
    3. self-attention over patch sequence
    4. MSUnit + P2D-FFN for period-aware 2D variation
    5. Intersect-Attention injects future exogenous variables
    6. forecast head outputs target sequence
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.enc_in = configs.enc_in
        self.c_out = getattr(configs, "c_out", configs.enc_in)

        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.dropout = configs.dropout

        self.patch_len = configs.patch_len
        self.patch_num = math.ceil(self.seq_len / self.patch_len)

        self.exog_dim = getattr(configs, "exog_dim", None)
        if self.exog_dim is None:
            self.exog_dim = getattr(configs, "mark_dim", 0)

        self.use_future_exog = getattr(configs, "use_future_exog", True)

        self.EPS = 1e-5

        self.patch_embedding = PatchEmbedding(
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            d_model=self.d_model,
            d_ff=self.d_ff,
            enc_in=self.enc_in
        )

        self.global_token = nn.Parameter(
            torch.randn(1, self.enc_in, 1, self.d_model) * 0.02
        )

        if self.use_future_exog and self.exog_dim > 0:
            self.exog_embedding = ExogenousEmbedding(
                pred_len=self.pred_len,
                exog_dim=self.exog_dim,
                d_model=self.d_model,
                d_ff=self.d_ff
            )
        else:
            self.exog_embedding = None

        top_k = getattr(configs, "top_k", 3)

        self.encoder = nn.ModuleList([
            EncoderLayer(
                d_model=self.d_model,
                d_ff=self.d_ff,
                n_heads=self.n_heads,
                top_k=top_k,
                dropout=self.dropout
            )
            for _ in range(self.e_layers)
        ])

        self.head = ForecastHead(
            patch_num=self.patch_num,
            d_model=self.d_model,
            d_ff=self.d_ff,
            pred_len=self.pred_len,
            c_out=self.c_out
        )

    def _get_future_exog(self, x_mark_dec):
        """
        Use the future known covariates over prediction horizon.
        x_mark_dec: [B, label_len + pred_len, M] or [B, pred_len, M]
        """
        if x_mark_dec is None or self.exog_embedding is None:
            return None

        if x_mark_dec.shape[1] >= self.pred_len:
            x_future = x_mark_dec[:, -self.pred_len:, :]
        else:
            raise ValueError(
                f"x_mark_dec length should be >= pred_len, "
                f"but got {x_mark_dec.shape[1]} and pred_len={self.pred_len}."
            )

        return self.exog_embedding(x_future)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        x_enc: [B, seq_len, enc_in]
        x_mark_dec: [B, label_len + pred_len, exog_dim]
        """

        # RevIN-style normalization
        means = x_enc.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + self.EPS
        ).detach()

        x = (x_enc - means) / stdev

        # Patch embedding: [B, C, P, D]
        h = self.patch_embedding(x)

        B = h.shape[0]

        # Add global token: [B, C, 1, D]
        g = self.global_token.repeat(B, 1, 1, 1)

        # [B, C, P+1, D]
        h = torch.cat([g, h], dim=2)

        # Future exogenous tokens: [B, M, D]
        exog_token = self._get_future_exog(x_mark_dec)

        for layer in self.encoder:
            h = layer(h, exog_token)

        # [B, pred_len, C]
        y = self.head(h)

        # De-normalization
        means = means[:, :, :self.c_out]
        stdev = stdev[:, :, :self.c_out]

        y = y * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)

        return y

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]
