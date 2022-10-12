import torch
import torch.nn as nn
import torch.nn.functional as F
from module.mp_encoder import Mp_encoder


def mask(x, mask_rate=0.5, noise=0.05):
    num_nodes = x.size(0)
    perm = torch.randperm(num_nodes, device=x.device)
    num_mask_nodes = int(mask_rate * num_nodes)

    # random masking
    # num_mask_nodes = int(mask_rate * num_nodes)
    mask_nodes = perm[: num_mask_nodes]
    keep_nodes = perm[num_mask_nodes:]

    num_noise_nodes = int(noise * num_mask_nodes)
    # 长度为打乱1354
    perm_mask = torch.randperm(num_mask_nodes, device=x.device)
    # 在1354的基础上随机选择1258
    token_nodes = mask_nodes[perm_mask[: int((1 - noise) * num_mask_nodes)]]
    # 1354减去1258剩下的67个
    noise_nodes = mask_nodes[perm_mask[-int(noise * num_mask_nodes):]]
    # 2708中随机选择67个替换上面的67个
    noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

    return token_nodes, noise_nodes, noise_to_be_chosen, mask_nodes


def sce_loss(x, y, alpha):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


class MAE(nn.Module):
    def __init__(
            self,
            use_data,
            hidden_dim,
            feat_drop,
            attn_drop,
            rate,
            noise,
            alpha
    ):
        super(MAE, self).__init__()

        self.hidden_dim = hidden_dim
        self.fc = nn.ModuleDict({
            n_type: nn.Linear(
                use_data[n_type].x.shape[1],
                hidden_dim,
                bias=True
            )
            for n_type in use_data.use_nodes
        })

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

        self.mp = Mp_encoder(use_data, hidden_dim, attn_drop, hidden_dim)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, hidden_dim))

        self.reset_parameter()

        self.rate = rate
        self.noise = noise
        self.alpha = alpha

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.PReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, use_data[use_data.main_node].x.shape[1])
        )

    def reset_parameter(self):
        for fc in self.fc.values():
            nn.init.xavier_normal_(fc.weight, gain=1.414)

    def forward(self, data):
        h = {}
        for n_type in data.use_nodes:
            h[n_type] = F.elu(
                self.feat_drop(
                    self.fc[n_type](data[n_type].x)
                )
            )
        token_nodes, noise_nodes, \
        noise_to_be_chosen, mask_nodes = mask(h[data.main_node],
                                              mask_rate=self.rate,
                                              noise=self.noise)
        h[data.main_node][token_nodes] = 0.0
        if self.noise > 0:
            h[data.main_node][noise_nodes] = h[data.main_node][noise_to_be_chosen]
        else:
            h[data.main_node][noise_nodes] = 0.0
        h[data.main_node][token_nodes] += self.enc_mask_token

        z_mp = self.mp(h, data)
        de_x = {data.main_node: z_mp}
        decoder = self.mlp(de_x[data.main_node])
        loss = sce_loss(decoder[token_nodes], data[data.main_node].x[token_nodes], self.alpha)
        return loss

    def get_embeds(self, data):
        h = {}
        for n_type in data.use_nodes:
            h[n_type] = F.elu(
                self.feat_drop(
                    self.fc[n_type](data[n_type].x)
                )
            )
        z_mp = self.mp(h, data)
        return z_mp.detach()
