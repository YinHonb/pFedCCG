import numpy as np
import torch

def _ndcg_at_k(hits, k):
    hits = np.asarray(hits, dtype=np.float32)[:k]
    if hits.size == 0:
        return 0.0
    denom = np.log2(np.arange(2, hits.size + 2))
    dcg = np.sum(hits / denom)

    ideal_hits = np.ones(int(min(np.sum(hits), k)), dtype=np.float32)
    if ideal_hits.size == 0:
        return 0.0
    ideal_denom = np.log2(np.arange(2, ideal_hits.size + 2))
    idcg = np.sum(ideal_hits / ideal_denom)
    return float(dcg / max(idcg, 1e-12))


def _hr_at_k(hits, k):
    hits = np.asarray(hits, dtype=np.float32)[:k]
    return float(np.sum(hits) > 0.0)


@torch.no_grad()
def compute_local_test_ranking_per_client(
    net,
    ml_meta,
    client_id,
    device="cpu",
    Ks=(5, 10, 20),
    num_neg=100,
    seed=0,
    max_users=None,
):
    net.eval()
    rng = np.random.RandomState(int(seed))

    n_items = int(ml_meta["n_items"])
    user_train_items = ml_meta["user_train_items"]
    client_pos = ml_meta["client_test_pos_items"][client_id]

    user_list = list(client_pos.keys())
    if max_users is not None and len(user_list) > max_users:
        rng.shuffle(user_list)
        user_list = user_list[:max_users]

    sum_recall = {K: 0.0 for K in Ks}
    sum_ndcg = {K: 0.0 for K in Ks}
    sum_hr = {K: 0.0 for K in Ks}
    n_eval = 0

    for u in user_list:
        pos_set = client_pos.get(int(u), None)
        if pos_set is None or len(pos_set) == 0:
            continue

        train_set = user_train_items[int(u)]

        negs = []
        tried = 0
        max_tried = int(num_neg) * 80
        while len(negs) < int(num_neg) and tried < max_tried:
            j = int(rng.randint(0, n_items))
            tried += 1
            if (j in train_set) or (j in pos_set):
                continue
            negs.append(j)
        if len(negs) == 0:
            continue

        pos_list = list(pos_set)
        cand_items = pos_list + negs

        u_t = torch.full((len(cand_items),), int(u), dtype=torch.long, device=device)
        i_t = torch.tensor(cand_items, dtype=torch.long, device=device)
        scores = net(u_t, i_t).view(-1).detach().cpu().numpy()

        order = np.argsort(-scores)
        ranked_items = [cand_items[idx] for idx in order]
        hits = np.array([1.0 if it in pos_set else 0.0 for it in ranked_items], dtype=np.float32)

        n_pos = len(pos_set)
        for K in Ks:
            topk_hits = hits[:K]
            recall_k = float(np.sum(topk_hits) / max(n_pos, 1))
            ndcg_k = _ndcg_at_k(hits, K)
            hr_k = _hr_at_k(hits, K)

            sum_recall[K] += recall_k
            sum_ndcg[K] += ndcg_k
            sum_hr[K] += hr_k

        n_eval += 1

    metrics = {}
    for K in Ks:
        metrics[f"Recall@{K}"] = float(sum_recall[K] / max(n_eval, 1))
        metrics[f"NDCG@{K}"] = float(sum_ndcg[K] / max(n_eval, 1))
        metrics[f"HR@{K}"] = float(sum_hr[K] / max(n_eval, 1))

    return metrics, int(n_eval)
