import torch
import numpy as np
import torch.optim as optim
import copy

from evaluate.evaluate_recsys import compute_local_test_ranking_per_client
from Data import movielens_dataset
from utils import optimize_W_l2

from model.movielens_ncf import NCF


class pFedCCG_recsys(object):
    def __init__(self, args, cfg=None):
        self.args = args
        self.cfg = cfg
        self.node_num = int(args.n_parties)

        dev = getattr(args, "device", None)
        if dev is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev)

        assert args.dataset in ["ml100k", "ml1m", "ml20m"], "MovieLens only: ml100k / ml1m / ml20m"
        self.is_recsys = True

        self.pos_threshold = float(getattr(args, "pos_threshold", 4))
        self.seed = self.args.init_seed
        self._rng = np.random.RandomState(self.seed)
        outs = movielens_dataset.dataset_read(
            dataset=args.dataset,
            base_path=args.datadir,
            batch_size=args.batch_size,
            n_parties=args.n_parties,
            partition=getattr(args, "partition", "schemeA"),
            beta=self.args.beta,
            skew_class=getattr(args, "skew_class", 0),
            seed=self.args.init_seed,
            pos_threshold=self.pos_threshold,
        )

        (self.train_dataloaders, self.val_dataloaders, self.test_dataloaders,
         self.net_dataidx_map, self.traindata_cls_counts, self.data_distributions,
         self.js_matrix, self.ml_meta) = outs

        print("js_matrix: ")
        for i in range(self.node_num):
            print(self.js_matrix[i])

        self.n_users = int(self.ml_meta["n_users"])
        self.n_items = int(self.ml_meta["n_items"])
        print("self.n_users", self.n_users)
        print("self.n_items", self.n_items)
        self.user_train_items = self.ml_meta["user_train_items"]

        assert args.recsys_model in ["mf", "ncf"], "Use --recsys_model mf or ncf"

        def _parse_layers(x, default=(128, 64, 32)):
            if x is None:
                return default
            if isinstance(x, (list, tuple)):
                return tuple(int(v) for v in x)
            if isinstance(x, str):
                s = x.strip()
                if len(s) == 0:
                    return default
                return tuple(int(v) for v in s.replace(" ", "").split(",") if len(v) > 0)
            return default

        mlp_layers = _parse_layers(getattr(args, "ncf_mlp_layers", None), default=(128, 64, 32))
        ncf_dropout = float(getattr(args, "ncf_dropout", 0.0))
        ncf_use_bias = bool(getattr(args, "ncf_use_bias", True))

        def build_model():
            return NCF(
                self.n_users,
                self.n_items,
                dim=int(args.embed_dim),
                mlp_layers=mlp_layers,
                dropout=ncf_dropout,
                use_bias=ncf_use_bias,
            )

        self.init_model = build_model().to(self.device)
        self.init_parameters = self.init_model.state_dict()

        self.model = [build_model().to(self.device) for _ in range(self.node_num)]
        for m in self.model:
            m.load_state_dict(self.init_parameters)

        total_data_points = sum([len(self.net_dataidx_map[k]) for k in range(self.node_num)])
        self.p_vector = np.array(
            [len(self.net_dataidx_map[k]) / max(total_data_points, 1) for k in range(self.node_num)],
            dtype=np.float64
        )

        self.W = optimize_W_l2(self.p_vector, self.data_distributions, 2)

        Ks = getattr(args, "Ks", (5, 10, 20, 30))
        self.Ks = tuple(Ks)

        self.best_recall_list = {K: [0.0] * self.node_num for K in self.Ks}
        self.best_ndcg_list = {K: [0.0] * self.node_num for K in self.Ks}
        self.best_hr_list = {K: [0.0] * self.node_num for K in self.Ks}


    def _sample_neg_items(self, users: torch.Tensor) -> torch.Tensor:
        users_np = users.detach().cpu().numpy()
        neg = np.empty_like(users_np)
        for idx, u in enumerate(users_np):
            pos_set = self.user_train_items[int(u)]
            tried = 0
            while True:
                j = self._rng.randint(0, self.n_items)
                tried += 1
                if j not in pos_set:
                    neg[idx] = j
                    break
                if tried > 2000:
                    neg[idx] = j
                    break
        return torch.from_numpy(neg).long().to(users.device)


    def step(self, now_time):
        self.W = optimize_W_l2(self.p_vector, self.data_distributions, 0.4 + 0.2/(1+np.exp((25-now_time)/10)))
        self.model_evaluate(now_time)
        self.local_train()
        self.model_evaluate(now_time)
        self.aggregation_model()

    def local_train(self):
        for node in range(self.node_num):
            loader = self.train_dataloaders[node]
            if loader is None:
                print(f"[Train] Client {node} | skip empty train loader.")
                continue

            net = self.model[node]

            optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.reg)

            net.train()

            for _epoch in range(self.args.epochs):
                for u, i_pos in loader:
                    u = u.to(self.device).long()
                    i_pos = i_pos.to(self.device).long()
                    i_neg = self._sample_neg_items(u)
                    optimizer.zero_grad()
                    s_pos = net(u, i_pos).view(-1)
                    s_neg = net(u, i_neg).view(-1)
                    loss = -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-12).mean()
                    loss.backward()
                    optimizer.step()

    def _is_float(self, t) -> bool:
        return torch.is_tensor(t) and torch.is_floating_point(t)

    def aggregation_model(self):
        models = {k: self.model[k] for k in range(self.node_num)}
        states = {k: models[k].state_dict() for k in models.keys()}

        for client_id in models.keys():
            w_vec = np.asarray(self.W[client_id], dtype=np.float64).copy()
            s = float(np.sum(w_vec))
            if (not np.isfinite(s)) or (s <= 0.0):
                w_vec[:] = 0.0
                w_vec[client_id] = 1.0
            else:
                w_vec /= s

            local_st = states[client_id]
            tmp = copy.deepcopy(local_st)

            for key in list(tmp.keys()):
                t = tmp[key]
                if self._is_float(t):
                    tmp[key] = torch.zeros_like(t)
                else:
                    tmp[key] = t.clone() if torch.is_tensor(t) else t

            for neighbor_id in models.keys():
                w = float(w_vec[neighbor_id])
                if w == 0.0:
                    continue
                st = states[neighbor_id]
                for key in list(tmp.keys()):
                    if self._is_float(st[key]):
                        tmp[key] += st[key] * w

            models[client_id].load_state_dict(tmp, strict=True)

    def model_evaluate(self, step_num):
        num_neg = int(getattr(self.args, "num_neg", 100))
        max_users = getattr(self.args, "ranking_max_users", None)

        for node in range(self.node_num):
            net = self.model[node]

            per_rank, per_n_users = compute_local_test_ranking_per_client(
                net,
                self.ml_meta,
                client_id=node,
                device=self.device,
                Ks=self.Ks,
                num_neg=num_neg,
                seed=int(getattr(self.args, "seed", 0)),
                max_users=max_users
            )

            for K in self.Ks:
                rK = per_rank.get(f"Recall@{K}", 0.0)
                nK = per_rank.get(f"NDCG@{K}", 0.0)
                hK = per_rank.get(f"HR@{K}", 0.0)

                if rK > self.best_recall_list[K][node]:
                    self.best_recall_list[K][node] = rK
                if nK > self.best_ndcg_list[K][node]:
                    self.best_ndcg_list[K][node] = nK
                if hK > self.best_hr_list[K][node]:
                    self.best_hr_list[K][node] = hK

            rank_str = " | ".join([
                f"Round {step_num}: R@{K}:{per_rank.get(f'Recall@{K}', 0.0):.4f},"
                f"Round {step_num}: N@{K}:{per_rank.get(f'NDCG@{K}', 0.0):.4f},"
                f"Round {step_num}: HR@{K}:{per_rank.get(f'HR@{K}', 0.0):.4f}"
                for K in self.Ks
            ])

            print(f">> Round {step_num}: Client {node} | Rank({per_n_users}u): {rank_str}")

        for K in self.Ks:
            r_arr = np.array(self.best_recall_list[K], dtype=np.float64)
            n_arr = np.array(self.best_ndcg_list[K], dtype=np.float64)
            h_arr = np.array(self.best_hr_list[K], dtype=np.float64)

            print(f">> Best@{K} Per-client: "
                  f"Round {step_num}: Recall {float(r_arr.mean()):.4f} ({float(r_arr.std()):.4f}) | "
                  f"Round {step_num}: NDCG {float(n_arr.mean()):.4f} ({float(n_arr.std()):.4f}) | "
                  f"Round {step_num}: HR {float(h_arr.mean()):.4f} ({float(h_arr.std()):.4f})")
