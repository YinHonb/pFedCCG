import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import optimize_W_l2
from Data import aya_dataset
from model.aya_lora_model import build_lora_model


@torch.no_grad()
def _extract_trainable_params_to_gpu(model, device):
    out = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            out[name] = p.detach().clone().to(device)
    return out


@torch.no_grad()
def _load_trainable_params_from_device(model, state_dict):
    name2p = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for n, w in state_dict.items():
        if n in name2p:
            name2p[n].data.copy_(w.data)


class pFedCCG_SFT(object):
    def __init__(self, args, cfg=None):
        self.args = args
        self.cfg = cfg
        self.node_num = int(args.n_parties)

        # 1) tokenizer + LoRA model
        self.tokenizer, self.model = build_lora_model(
            model_name_or_path=args.model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=getattr(args, "trust_remote_code", True),
            lora_r=getattr(args, "lora_r", 8),
            lora_alpha=getattr(args, "lora_alpha", 16),
            lora_dropout=getattr(args, "lora_dropout", 0.05),
            target_modules=getattr(args, "lora_target_modules", None),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("param device:", next(self.model.parameters()).device)

        (
            self.train_dataloaders,
            self.val_dataloaders,
            self.test_dataloaders,
            self.global_test_loader,
            self.net_dataidx_map,
            self.lang_counts,
            self.data_distributions,
            self.js_matrix,
            meta,
        ) = aya_dataset.dataset_read_aya(
            tokenizer=self.tokenizer,
            batch_size=int(getattr(args, "batch_size", 1)),
            n_parties=int(args.n_parties),
            top_k_langs=int(getattr(args, "aya_top_k_langs", 10)),
            max_length=int(getattr(args, "llm_max_length", 512)),
            seed=int(getattr(args, "init_seed", 42)),
            verbose=True,

            train_per_lang=int(getattr(args, "aya_train_per_lang", 1000)),
            test_per_lang=int(getattr(args, "aya_test_per_lang", 50)),
            oversample_factor=float(getattr(args, "aya_oversample_factor", 3.0)),

            k_langs_per_client=int(getattr(args, "aya_k_langs_per_client", 3)),
            min_require_size=int(getattr(args, "aya_min_require_size", 10)),

            num_workers=int(getattr(args, "num_workers", 0)),
        )

        self.meta = meta
        self.n_langs = int(meta["n_langs"])
        self.client_lang_weights = np.asarray(meta["client_lang_weights"], dtype=np.float64)

        print(">> Selected languages:", self.meta["selected_lang_codes"])
        print(">> client_lang_weights shape:", self.client_lang_weights.shape)

        total_data_points = sum(len(self.net_dataidx_map[k]) for k in range(self.node_num))
        self.p_vector = np.array(
            [len(self.net_dataidx_map[k]) / max(total_data_points, 1) for k in range(self.node_num)],
            dtype=np.float32,
        )

        init_lora = _extract_trainable_params_to_gpu(self.model, self.device)
        self.init_lora = copy.deepcopy(init_lora)
        self.client_lora = [copy.deepcopy(init_lora) for _ in range(self.node_num)]

        self.W = optimize_W_l2(self.p_vector, self.data_distributions, 0.8)


    def step(self, now_time):
        self.round = now_time
        self.W = optimize_W_l2(self.p_vector, self.data_distributions, 0.4 + 0.2 / (1 + np.exp((15 - now_time) / 7)))
        self.model_evaluate(now_time)
        self.local_train_all_clients()
        self.model_evaluate(now_time)
        self.aggregation_model()

    def local_train_all_clients(self):
        for node in range(self.node_num):
            self.local_train_one_client(node)

    def local_train_one_client(self, node):
        print(f"[Local Train] client={node}")

        _load_trainable_params_from_device(self.model, self.client_lora[node])
        self.model.train()

        optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(getattr(self.args, "llm_lr", 2e-4)),
            weight_decay=float(getattr(self.args, "reg", 0.0)),
        )

        grad_accum = int(getattr(self.args, "grad_accum", 1))
        max_grad_norm = float(getattr(self.args, "max_grad_norm", 1.0))
        keep_prob = float(getattr(self.args, "epoch_sample_ratio", 1.0))

        dataloader = self.train_dataloaders[node]
        step = 0
        optimizer.zero_grad(set_to_none=True)

        for _epoch in range(int(getattr(self.args, "epochs", 1))):
            for batch in dataloader:
                if np.random.rand() > keep_prob:
                    continue

                batch = {k: v.to(self.device) for k, v in batch.items()
                         if k in ["input_ids", "attention_mask", "labels"]}

                out = self.model(**batch)
                loss = out.loss / grad_accum
                loss.backward()

                step += 1
                if step % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        max_grad_norm,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            if step % grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        self.client_lora[node] = _extract_trainable_params_to_gpu(self.model, self.device)

    def aggregation_model(self):
        new_client_lora = [None for _ in range(self.node_num)]
        with torch.no_grad():
            for client_id in range(self.node_num):
                w = np.asarray(self.W[client_id], dtype=np.float64)
                w = w / (w.sum() + 1e-12)

                agg = {}
                for k in self.client_lora[0].keys():
                    agg[k] = torch.zeros_like(
                        self.client_lora[0][k],
                        device=self.device,
                        dtype=self.client_lora[0][k].dtype,
                    )

                for neighbor_id in range(self.node_num):
                    if w[neighbor_id] == 0:
                        continue
                    weight = float(w[neighbor_id])
                    neigh = self.client_lora[neighbor_id]
                    for k in agg.keys():
                        if neigh[k].device != self.device:
                            neigh[k] = neigh[k].to(self.device)
                        agg[k].add_(neigh[k] * weight)

                new_client_lora[client_id] = agg

        self.client_lora = new_client_lora

    def model_evaluate(self, step_num):
        log_path = "evaluation_log.txt"

        top_m = int(getattr(self.args, "eval_top_m", 3))
        cvar_rho = float(getattr(self.args, "eval_cvar_rho", 0.2))

        topks = getattr(self.args, "eval_topks", [1, 3, 5, 7, 9, 11, 13, 15, 20, 30])
        if isinstance(topks, str):
            topks = [int(x) for x in topks.replace(" ", "").split(",") if len(x) > 0]
        elif isinstance(topks, (int, np.integer)):
            topks = [int(topks)]
        else:
            topks = [int(x) for x in list(topks)]
        topks = sorted(set([k for k in topks if k >= 1]))
        if 1 not in topks:
            topks = [1] + topks
        if len(topks) == 0:
            topks = [1, 5]

        mrr_k = int(getattr(self.args, "eval_mrr_k", max(topks)))
        Kmax = int(max(max(topks), mrr_k))

        p = np.asarray(self.p_vector, dtype=np.float64)
        p = p / (p.sum() + 1e-12)

        def _wmean(x, w):
            x = np.asarray(x, dtype=np.float64)
            w = np.asarray(w, dtype=np.float64)
            return float(np.nansum(w * x))

        def _wstd(x, w):
            x = np.asarray(x, dtype=np.float64)
            w = np.asarray(w, dtype=np.float64)
            m = _wmean(x, w)
            var = np.nansum(w * (x - m) ** 2)
            return float(np.sqrt(max(var, 0.0)))

        def _weighted_std(values, weights):
            values = np.asarray(values, dtype=np.float64)
            weights = np.asarray(weights, dtype=np.float64)
            weights = weights / (weights.sum() + 1e-12)
            m = float(np.sum(weights * values))
            var = float(np.sum(weights * (values - m) ** 2))
            return float(np.sqrt(max(var, 0.0)))

        def _cvar(values, weights, rho):
            values = np.asarray(values, dtype=np.float64)
            weights = np.asarray(weights, dtype=np.float64)
            weights = weights / (weights.sum() + 1e-12)
            rho = float(np.clip(rho, 1e-6, 1.0))

            order = np.argsort(-values)
            v = values[order]
            w = weights[order]

            take = rho
            acc = 0.0
            for vi, wi in zip(v, w):
                if take <= 0:
                    break
                t = wi if wi <= take else take
                acc += float(vi) * float(t)
                take -= float(t)

            return float(acc / rho)

        @torch.no_grad()
        def _eval_by_lang_multi_topk(net, dataloader, n_langs, device, topks, Kmax, lang_key="lang_id"):
            net.eval()
            L = int(n_langs)

            per_lang_loss_sum = torch.zeros(L, device=device, dtype=torch.float64)
            per_lang_loss_sum_sq = torch.zeros(L, device=device, dtype=torch.float64)
            per_lang_tok = torch.zeros(L, device=device, dtype=torch.float64)

            per_lang_correct = {k: torch.zeros(L, device=device, dtype=torch.float64) for k in topks}
            per_lang_mrr_sum = torch.zeros(L, device=device, dtype=torch.float64)

            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                if lang_key not in batch:
                    raise KeyError(f"batch has no '{lang_key}'. keys={list(batch.keys())}")
                lang_id = batch[lang_key].to(device).long().view(-1)

                out = net(input_ids=input_ids, attention_mask=attn)
                logits = out.logits

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                B, Tm1, V = shift_logits.shape
                mask = (shift_labels != -100)
                mask_f = mask.float()

                # token-level NLL
                loss_flat = F.cross_entropy(
                    shift_logits.view(-1, V),
                    shift_labels.view(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view(B, Tm1)

                loss_sum_b = (loss_flat * mask_f).sum(dim=1).to(torch.float64)  # [B]
                loss_sq_sum_b = ((loss_flat * mask_f) ** 2).sum(dim=1).to(torch.float64)  # [B]
                tok_b = mask.sum(dim=1).to(torch.float64)  # [B]

                per_lang_loss_sum.index_add_(0, lang_id, loss_sum_b)
                per_lang_loss_sum_sq.index_add_(0, lang_id, loss_sq_sum_b)
                per_lang_tok.index_add_(0, lang_id, tok_b)

                # TopK once
                K = int(min(max(int(Kmax), 1), V))
                top_idx = shift_logits.topk(k=K, dim=-1).indices
                match = (top_idx == shift_labels.unsqueeze(-1)) & mask.unsqueeze(-1)

                # Acc@k
                for k in topks:
                    kk = int(min(k, K))
                    hitk = match[:, :, :kk].any(dim=-1) & mask
                    correctk_b = hitk.sum(dim=1).to(torch.float64)
                    per_lang_correct[k].index_add_(0, lang_id, correctk_b)

                # MRR@K (approx)
                hit_any = match.any(dim=-1)
                pos = match.float().argmax(dim=-1) + 1.0
                mrr_tok = hit_any.float() / pos
                mrr_sum_b = (mrr_tok * mask_f).sum(dim=1).to(torch.float64)
                per_lang_mrr_sum.index_add_(0, lang_id, mrr_sum_b)

            denom = torch.clamp(per_lang_tok, min=1.0)
            per_lang_nll = (per_lang_loss_sum / denom).cpu().numpy()
            per_lang_mrr = (per_lang_mrr_sum / denom).cpu().numpy()
            per_lang_acck = {k: (per_lang_correct[k] / denom).cpu().numpy() for k in topks}

            return {
                "per_lang_nll": per_lang_nll,
                "per_lang_tok": per_lang_tok.cpu().numpy(),
                "per_lang_loss_sum": per_lang_loss_sum.cpu().numpy(),
                "per_lang_loss_sum_sq": per_lang_loss_sum_sq.cpu().numpy(),
                "per_lang_acck": per_lang_acck,
                "per_lang_mrr": per_lang_mrr,
                "Kmax": int(Kmax),
            }

        N = int(self.node_num)

        # NLL
        if not hasattr(self, "best_weighted_nll"):
            self.best_weighted_nll = [float("inf")] * N
            self.best_weighted_round = [-1] * N

        # Acc@k best
        need_reset_acck = (
                (not hasattr(self, "best_weighted_acck"))
                or (not isinstance(getattr(self, "best_weighted_acck", None), dict))
                or (set(self.best_weighted_acck.keys()) != set(topks))
        )
        if need_reset_acck:
            self.best_weighted_acck = {k: [-float("inf")] * N for k in topks}
            self.best_weighted_acck_round = {k: [-1] * N for k in topks}

        # MRR best
        if not hasattr(self, "best_weighted_mrr"):
            self.best_weighted_mrr = [-float("inf")] * N
            self.best_weighted_mrr_round = [-1] * N

        # NLL-side extras (lower better)
        if not hasattr(self, "best_pers_std_nll"):
            self.best_pers_std_nll = [float("inf")] * N
            self.best_pers_std_round = [-1] * N
        if not hasattr(self, "best_topworst_nll"):
            self.best_topworst_nll = [float("inf")] * N
            self.best_topworst_round = [-1] * N
        if not hasattr(self, "best_cvar_nll"):
            self.best_cvar_nll = [float("inf")] * N
            self.best_cvar_round = [-1] * N

        # Acc tail best (higher better)
        def _ensure_best_dict(name):
            cur = getattr(self, name, None)
            if (not isinstance(cur, dict)) or (set(cur.keys()) != set(topks)):
                setattr(self, name, {k: [-float("inf")] * N for k in topks})
                setattr(self, name + "_round", {k: [-1] * N for k in topks})

        _ensure_best_dict("best_topworst_acck")
        _ensure_best_dict("best_cvar_acck")

        # MRR tail best (higher better)
        if not hasattr(self, "best_topworst_mrr"):
            self.best_topworst_mrr = [-float("inf")] * N
            self.best_topworst_mrr_round = [-1] * N
        if not hasattr(self, "best_cvar_mrr"):
            self.best_cvar_mrr = [-float("inf")] * N
            self.best_cvar_mrr_round = [-1] * N

        best_ppl_list = []
        best_acck_list = {k: [] for k in topks}
        best_mrr_list = []

        best_std_list = []
        best_topworst_nll_list = []
        best_cvar_nll_list = []

        best_topworst_acck_list = {k: [] for k in topks}
        best_cvar_acck_list = {k: [] for k in topks}
        best_topworst_mrr_list = []
        best_cvar_mrr_list = []

        with open(log_path, "a", encoding="utf-8") as f:
            for node in range(N):
                _load_trainable_params_from_device(self.model, self.client_lora[node])
                self.model.eval()

                by = _eval_by_lang_multi_topk(
                    self.model,
                    self.global_test_loader,
                    n_langs=self.n_langs,
                    device=self.device,
                    topks=topks,
                    Kmax=Kmax,
                    lang_key="lang_id",
                )

                per_lang_nll = np.asarray(by["per_lang_nll"], dtype=np.float64)
                per_lang_tok = np.asarray(by["per_lang_tok"], dtype=np.float64)
                per_lang_acck = {k: np.asarray(by["per_lang_acck"][k], dtype=np.float64) for k in topks}
                per_lang_mrr = np.asarray(by["per_lang_mrr"], dtype=np.float64)

                w_full = np.asarray(self.client_lang_weights[node], dtype=np.float64)
                mask = (w_full > 0) & (per_lang_tok > 0)

                if mask.sum() == 0:
                    pers_nll = float(np.nanmean(per_lang_nll))
                    pers_ppl = float(np.exp(pers_nll))
                    pers_acck = {k: float(np.nanmean(per_lang_acck[k])) for k in topks}
                    pers_mrr = float(np.nanmean(per_lang_mrr))

                    pers_std_nll = float("nan")
                    topworst_nll = float("nan")
                    cvar_nll = float("nan")

                    topworst_acck = {k: float("nan") for k in topks}
                    cvar_acck = {k: float("nan") for k in topks}
                    topworst_mrr = float("nan")
                    cvar_mrr = float("nan")
                else:
                    idx = np.where(mask)[0]
                    w_sup = w_full[idx]
                    w2 = w_sup / (w_sup.sum() + 1e-12)

                    nll_sup = per_lang_nll[idx]
                    acck_sup = {k: per_lang_acck[k][idx] for k in topks}
                    mrr_sup = per_lang_mrr[idx]

                    # Pers metrics
                    pers_nll = float(np.sum(w2 * nll_sup))
                    pers_ppl = float(np.exp(pers_nll))
                    pers_acck = {k: float(np.sum(w2 * acck_sup[k])) for k in topks}
                    pers_mrr = float(np.sum(w2 * mrr_sup))

                    pers_std_nll = _weighted_std(nll_sup, w2)

                    m = int(min(max(top_m, 1), idx.shape[0]))
                    topm_order = np.argsort(-w_sup)[:m]
                    topworst_nll = float(np.max(nll_sup[topm_order]))
                    cvar_nll = _cvar(nll_sup, w2, cvar_rho)

                    topworst_acck = {}
                    cvar_acck = {}
                    for k in topks:
                        topworst_acck[k] = float(np.min(acck_sup[k][topm_order]))
                        cvar_acck[k] = 1.0 - _cvar(1.0 - acck_sup[k], w2, cvar_rho)

                    topworst_mrr = float(np.min(mrr_sup[topm_order]))
                    cvar_mrr = 1.0 - _cvar(1.0 - mrr_sup, w2, cvar_rho)

                if np.isfinite(pers_nll) and pers_nll < self.best_weighted_nll[node]:
                    self.best_weighted_nll[node] = float(pers_nll)
                    self.best_weighted_round[node] = int(step_num)

                for k in topks:
                    if np.isfinite(pers_acck[k]) and pers_acck[k] > self.best_weighted_acck[k][node]:
                        self.best_weighted_acck[k][node] = float(pers_acck[k])
                        self.best_weighted_acck_round[k][node] = int(step_num)

                if np.isfinite(pers_mrr) and pers_mrr > self.best_weighted_mrr[node]:
                    self.best_weighted_mrr[node] = float(pers_mrr)
                    self.best_weighted_mrr_round[node] = int(step_num)

                if np.isfinite(pers_std_nll) and pers_std_nll < self.best_pers_std_nll[node]:
                    self.best_pers_std_nll[node] = float(pers_std_nll)
                    self.best_pers_std_round[node] = int(step_num)

                if np.isfinite(topworst_nll) and topworst_nll < self.best_topworst_nll[node]:
                    self.best_topworst_nll[node] = float(topworst_nll)
                    self.best_topworst_round[node] = int(step_num)

                if np.isfinite(cvar_nll) and cvar_nll < self.best_cvar_nll[node]:
                    self.best_cvar_nll[node] = float(cvar_nll)
                    self.best_cvar_round[node] = int(step_num)

                for k in topks:
                    if np.isfinite(topworst_acck[k]) and topworst_acck[k] > self.best_topworst_acck[k][node]:
                        self.best_topworst_acck[k][node] = float(topworst_acck[k])
                        self.best_topworst_acck_round[k][node] = int(step_num)

                    if np.isfinite(cvar_acck[k]) and cvar_acck[k] > self.best_cvar_acck[k][node]:
                        self.best_cvar_acck[k][node] = float(cvar_acck[k])
                        self.best_cvar_acck_round[k][node] = int(step_num)

                # MRR tail best (higher better)
                if np.isfinite(topworst_mrr) and topworst_mrr > self.best_topworst_mrr[node]:
                    self.best_topworst_mrr[node] = float(topworst_mrr)
                    self.best_topworst_mrr_round[node] = int(step_num)

                if np.isfinite(cvar_mrr) and cvar_mrr > self.best_cvar_mrr[node]:
                    self.best_cvar_mrr[node] = float(cvar_mrr)
                    self.best_cvar_mrr_round[node] = int(step_num)

                best_nll = float(self.best_weighted_nll[node])
                best_ppl = float(np.exp(best_nll)) if np.isfinite(best_nll) else float("inf")

                acck_str = " ".join([f"A@{k} {pers_acck[k]:.4f}" for k in topks])
                tw_acck_str = " ".join([f"TW_A@{k} {topworst_acck[k]:.4f}" for k in topks])
                cv_acck_str = " ".join([f"CV_A@{k} {cvar_acck[k]:.4f}" for k in topks])

                line = (
                    f">> Round {step_num} | Client {node} | "
                    f"CUR: NLL {pers_nll:.4f} PPL {pers_ppl:.4f} | "
                    f"{acck_str} | MRR@{Kmax} {pers_mrr:.4f} | "
                    f"StdNLL {pers_std_nll:.4f} TW_NLL(top{top_m}) {topworst_nll:.4f} CVaR_NLL@{cvar_rho:.2f} {cvar_nll:.4f} | "
                    f"{tw_acck_str} | {cv_acck_str} | "
                    f"TW_MRR {topworst_mrr:.4f} CVaR_MRR@{cvar_rho:.2f} {cvar_mrr:.4f} || "
                    f"BEST(NLL)@R{self.best_weighted_round[node]} {best_nll:.4f} (PPL {best_ppl:.4f}) | "
                    f"BEST(MRR@{Kmax})@R{self.best_weighted_mrr_round[node]} {self.best_weighted_mrr[node]:.4f}"
                )
                print(line)
                f.write(line + "\n")

                best_ppl_list.append(best_ppl)
                for k in topks:
                    best_acck_list[k].append(float(self.best_weighted_acck[k][node]))
                best_mrr_list.append(float(self.best_weighted_mrr[node]))

                best_std_list.append(float(self.best_pers_std_nll[node]))
                best_topworst_nll_list.append(float(self.best_topworst_nll[node]))
                best_cvar_nll_list.append(float(self.best_cvar_nll[node]))

                for k in topks:
                    best_topworst_acck_list[k].append(float(self.best_topworst_acck[k][node]))
                    best_cvar_acck_list[k].append(float(self.best_cvar_acck[k][node]))
                best_topworst_mrr_list.append(float(self.best_topworst_mrr[node]))
                best_cvar_mrr_list.append(float(self.best_cvar_mrr[node]))

            def _summ(name, arr_best):
                arr_best = np.asarray(arr_best, dtype=np.float64)
                mean_best = float(np.nanmean(arr_best))
                std_best = float(np.nanstd(arr_best))
                wmean_best = _wmean(arr_best, p)
                wstd_best = _wstd(arr_best, p)
                s = (
                    f">> Round {step_num} | [AVG {name}] "
                    f"Mean(BEST)={mean_best:.4f} Std(BEST)={std_best:.4f} || "
                    f"pMean(BEST)={wmean_best:.4f} pStd(BEST)={wstd_best:.4f}"
                )
                print(s)
                f.write(s + "\n")
                return mean_best

            mean_best_ppl = _summ("Personalized PPL", best_ppl_list)

            for k in topks:
                _summ(f"Personalized Acc@{k}", best_acck_list[k])
            _summ(f"Personalized MRR@{Kmax}", best_mrr_list)

            _summ("Personalized StdNLL", best_std_list)
            _summ(f"Personalized TopWorstNLL(top{top_m})", best_topworst_nll_list)
            _summ(f"Personalized CVaR_NLL@{cvar_rho:.2f}", best_cvar_nll_list)

            for k in topks:
                _summ(f"Personalized TopWorstAcc@{k}(top{top_m})", best_topworst_acck_list[k])
                _summ(f"Personalized CVaR_Acc@{k}@{cvar_rho:.2f}", best_cvar_acck_list[k])

            _summ(f"Personalized TopWorstMRR@{Kmax}(top{top_m})", best_topworst_mrr_list)
            _summ(f"Personalized CVaR_MRR@{cvar_rho:.2f}", best_cvar_mrr_list)

            sep = "=" * 80
            print(sep)
            f.write(sep + "\n")

        return float(mean_best_ppl)

    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
