import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, T, ch=128, ch_mult=[1, 2, 3, 4], attn=[2], num_res_blocks=2, dropout=0.1):
        super(UNet, self).__init__()
        self.T = T
        self.ch = ch

        self.time_embed = nn.Sequential(
            nn.Linear(T, ch),
            nn.ReLU(),
            nn.Linear(ch, ch * 2)  # ✅ fixed: match conv2 output channels
        )

        self.conv1 = nn.Conv2d(3, ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch * 2, kernel_size=3, padding=1)
        self.deconv = nn.ConvTranspose2d(ch * 2, ch, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(ch, 3, kernel_size=3, padding=1)

        self.act = nn.ReLU()

    def forward(self, x, t):
        t_embed = self._timestep_embedding(t, self.T)
        t_embed = self.time_embed(t_embed).unsqueeze(-1).unsqueeze(-1)
        x = self.act(self.conv1(x) + t_embed[:, :self.ch])
        x = self.act(self.conv2(x) + t_embed)
        x = self.act(self.deconv(x) + t_embed[:, :self.ch])
        x = self.out_conv(x)
        return x

    def _timestep_embedding(self, timesteps, dim):
        half = dim // 2
        emb = torch.exp(torch.arange(half, dtype=torch.float32, device=timesteps.device) * -torch.log(torch.tensor(10000.0)) / (half - 1))
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb
