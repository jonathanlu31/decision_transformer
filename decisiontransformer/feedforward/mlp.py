import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaInMLP(nn.Module):
    def __init__(self, num_hidden, w_in, w_h, w_out, num_freqs, weight_conditional=False, learn_sigma=False):
        super(AdaInMLP, self).__init__()
        D = 1
        self.cond_emb_dim = 2 * num_freqs * D + D
        self.fcs, self.affs = self._build_mlp(num_hidden, w_in, w_h, w_out, self.cond_emb_dim,
                                              weight_conditional, learn_sigma)
        self.embedder = FrequencyEmbedder(num_frequencies=num_freqs, max_freq_log2=8)
        self.weight_conditional = weight_conditional

    @staticmethod
    def _build_mlp(num_hidden, w_in, w_h, w_out, cond_emb_dim, weight_conditional, learn_sigma):
        fcs = nn.ModuleList()
        affs = nn.ModuleList()
        for i in range(num_hidden):
            in_dim = w_in * (1 + weight_conditional) if i == 0 else w_h
            out_dim = w_out * (1 + learn_sigma) if i == num_hidden - 1 else w_h
            lin = nn.Linear(in_dim, out_dim)
            aff = nn.Linear(cond_emb_dim, 2 * out_dim) if i < num_hidden - 1 else None
            fcs.append(lin)
            affs.append(aff)
            if i == num_hidden - 1:  # Final layer:
                fcs[-1].weight.data.zero_()
                fcs[-1].bias.data.zero_()
        return fcs, affs

    @staticmethod
    def _forward_adain(h, cond_emb, lin, aff):
        shift, scale = aff(cond_emb).chunk(2, dim=1)
        h_next = lin(h)
        h_next = F.layer_norm(h_next, [h_next.shape[-1]])
        if aff is not None:
            h_next = h_next * (1 + scale) + shift
        h_next = torch.relu(h_next)
        return h_next

    def forward(self, x, t, x_prev=None):
        cond_emb = self.embedder(t)
        if x_prev is not None and self.weight_conditional:  # conditional model
            h = torch.cat([x, x_prev], 1)
        else:
            h = x
        for i, (lin, aff) in enumerate(zip(self.fcs, self.affs)):
            if i < len(self.fcs) - 1:  # All layers but last:
                h = self._forward_adain(h, cond_emb, lin, aff)
            else:
                h = lin(h)
        return h

class FrequencyEmbedder(nn.Module):

    def __init__(self, num_frequencies, max_freq_log2):
        super().__init__()
        frequencies = 2 ** torch.linspace(0, max_freq_log2, steps=num_frequencies)
        self.register_buffer('frequencies', frequencies)

    def forward(self, x):
        # x should be of size (N,) or (N, D)
        N = x.size(0)
        if x.dim() == 1:  # (N,)
            x = x.unsqueeze(1)  # (N, D) where D=1
        x_unsqueezed = x.unsqueeze(-1).to('cuda', torch.float)  # (N, D, 1)
        scaled = self.frequencies.view(1, 1, -1) * x_unsqueezed  # (N, D, num_frequencies)
        s = torch.sin(scaled)
        c = torch.cos(scaled)
        embedded = torch.cat([s, c, x_unsqueezed], dim=-1).view(N, -1)  # (N, D * 2 * num_frequencies + D)
        return embedded