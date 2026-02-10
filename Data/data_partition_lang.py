import numpy as np
from collections import defaultdict


def calculate_js_divergence_matrix(data: np.ndarray) -> np.ndarray:
    data = data.astype(np.float64)
    num_clients = data.shape[0]
    out = np.zeros((num_clients, num_clients), dtype=np.float64)

    def kl(p, q):
        p = np.where(p == 0, 1e-12, p)
        q = np.where(q == 0, 1e-12, q)
        return np.sum(p * np.log2(p / q))

    def js_sim(p, q):
        p = p / (p.sum() + 1e-12)
        q = q / (q.sum() + 1e-12)
        m = 0.5 * (p + q)
        js = 0.5 * (kl(p, m) + kl(q, m))
        sim = 1.0 - js
        return float(np.clip(sim, 0.0, 1.0))

    for i in range(num_clients):
        for j in range(num_clients):
            out[i, j] = js_sim(data[i], data[j])
    return out


def build_client_language_sets(
    n_clients: int,
    langs: list,
    min_langs: int = 2,
    max_langs: int = 4,
    avg_langs: int = 3,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    n_langs = len(langs)

    total_slots = n_clients * avg_langs
    cover = max(2, int(round(total_slots / max(n_langs, 1))))

    client_langs = [set() for _ in range(n_clients)]

    for lang in langs:
        order = rng.permutation(n_clients).tolist()
        picked = []
        for c in order:
            if len(client_langs[c]) < max_langs:
                picked.append(c)
                if len(picked) >= cover:
                    break
        for c in picked:
            client_langs[c].add(lang)

    for c in range(n_clients):
        while len(client_langs[c]) < min_langs:
            client_langs[c].add(int(rng.choice(langs)))

    for c in range(n_clients):
        target = int(rng.integers(min_langs, max_langs + 1))
        while len(client_langs[c]) < target:
            client_langs[c].add(int(rng.choice(langs)))

    return [sorted(list(s)) for s in client_langs]


def partition_aya_by_language_overlap(
    lang_ids: np.ndarray,
    n_clients: int = 20,
    n_langs: int = 8,
    min_langs_per_client: int = 2,
    max_langs_per_client: int = 4,
    avg_langs_per_client: int = 3,
    seed: int = 42,
    verbose: bool = False,
    **kwargs,
):
    rng = np.random.default_rng(seed)
    N = len(lang_ids)
    langs = list(range(n_langs))

    client_lang_sets = build_client_language_sets(
        n_clients=n_clients,
        langs=langs,
        min_langs=min_langs_per_client,
        max_langs=max_langs_per_client,
        avg_langs=avg_langs_per_client,
        seed=seed,
    )

    clients_for_lang = defaultdict(list)
    for c in range(n_clients):
        for l in client_lang_sets[c]:
            clients_for_lang[l].append(c)

    idx_by_lang = {}
    for l in langs:
        idx_l = np.where(lang_ids == l)[0].copy()
        rng.shuffle(idx_l)
        idx_by_lang[l] = idx_l

    net_dataidx_map = {c: [] for c in range(n_clients)}
    total_assigned = np.zeros(n_clients, dtype=np.int64)

    for l in langs:
        cands = clients_for_lang[l]
        if len(cands) == 0:
            continue
        for idx in idx_by_lang[l]:
            c = cands[int(np.argmin(total_assigned[cands]))]
            net_dataidx_map[c].append(int(idx))
            total_assigned[c] += 1

    for c in range(n_clients):
        rng.shuffle(net_dataidx_map[c])

    lang_counts = np.zeros((n_clients, n_langs), dtype=np.int64)
    for c in range(n_clients):
        ids = np.array(net_dataidx_map[c], dtype=np.int64)
        if len(ids) == 0:
            continue
        lang_counts[c] = np.bincount(lang_ids[ids], minlength=n_langs)

    lang_distributions = lang_counts / (lang_counts.sum(axis=1, keepdims=True) + 1e-12)
    js_matrix = calculate_js_divergence_matrix(lang_counts.astype(np.float64))

    if verbose:
        print(f"[Aya] JS-matrix stats (1-JS similarity): min={js_matrix.min():.4f}, mean={js_matrix.mean():.4f}, max={js_matrix.max():.4f}")

    return net_dataidx_map, lang_counts, lang_distributions, js_matrix, client_lang_sets
