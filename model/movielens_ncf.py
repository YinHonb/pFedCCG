import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim: int = 64,
        mlp_layers=(128, 64, 32),
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super().__init__()
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.dim = int(dim)
        self.use_bias = bool(use_bias)

        # GMF
        self.user_emb_gmf = nn.Embedding(self.n_users, self.dim)
        self.item_emb_gmf = nn.Embedding(self.n_items, self.dim)

        # MLP
        self.user_emb_mlp = nn.Embedding(self.n_users, self.dim)
        self.item_emb_mlp = nn.Embedding(self.n_items, self.dim)

        layers = []
        in_dim = 2 * self.dim
        for h in mlp_layers:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(nn.ReLU(inplace=True))
            if dropout and float(dropout) > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            in_dim = int(h)
        self.mlp = nn.Sequential(*layers) if len(layers) > 0 else nn.Identity()
        mlp_out_dim = in_dim

        self.out = nn.Linear(self.dim + mlp_out_dim, 1)

        if self.use_bias:
            self.user_bias = nn.Embedding(self.n_users, 1)
            self.item_bias = nn.Embedding(self.n_items, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))
        else:
            self.user_bias = None
            self.item_bias = None
            self.global_bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_emb_gmf.weight, std=0.01)
        nn.init.normal_(self.item_emb_gmf.weight, std=0.01)
        nn.init.normal_(self.user_emb_mlp.weight, std=0.01)
        nn.init.normal_(self.item_emb_mlp.weight, std=0.01)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        if self.use_bias:
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
            with torch.no_grad():
                self.global_bias.zero_()

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        ug = self.user_emb_gmf(user)
        ig = self.item_emb_gmf(item)
        gmf = ug * ig

        um = self.user_emb_mlp(user)
        im = self.item_emb_mlp(item)
        x = torch.cat([um, im], dim=1)
        mlp_out = self.mlp(x)

        z = torch.cat([gmf, mlp_out], dim=1)
        y = self.out(z).squeeze(1)

        if self.use_bias:
            y = y + self.user_bias(user).squeeze(1) + self.item_bias(item).squeeze(1) + self.global_bias
        return y
