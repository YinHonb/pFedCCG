import os
import zipfile
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



def _download(url, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not os.path.exists(out_path):
        print(f"[MovieLens] downloading: {url}")
        urllib.request.urlretrieve(url, out_path)


def _extract(zip_path, root):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

def _load_ratings_ml100k(root):
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = os.path.join(root, "ml-100k.zip")
    folder = os.path.join(root, "ml-100k")
    if not os.path.exists(folder):
        _download(url, zip_path)
        _extract(zip_path, root)

    path = os.path.join(folder, "u.data")
    rows = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            u, i, r, ts = line.strip().split("\t")
            rows.append((int(u), int(i), float(r), int(ts)))
    rows = np.asarray(rows, dtype=np.float64)
    user = rows[:, 0].astype(np.int64)
    item = rows[:, 1].astype(np.int64)
    rating = rows[:, 2].astype(np.float32)
    ts = rows[:, 3].astype(np.int64)
    return user, item, rating, ts


def _load_ratings_ml1m(root):
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = os.path.join(root, "ml-1m.zip")
    folder = os.path.join(root, "ml-1m")
    if not os.path.exists(folder):
        _download(url, zip_path)
        _extract(zip_path, root)

    path = os.path.join(folder, "ratings.dat")
    rows = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            u, i, r, ts = line.strip().split("::")
            rows.append((int(u), int(i), float(r), int(ts)))
    rows = np.asarray(rows, dtype=np.float64)
    user = rows[:, 0].astype(np.int64)
    item = rows[:, 1].astype(np.int64)
    rating = rows[:, 2].astype(np.float32)
    ts = rows[:, 3].astype(np.int64)
    return user, item, rating, ts


def _load_ratings_ml20m(root, max_ratings=None, seed=0):
    url = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    zip_path = os.path.join(root, "ml-20m.zip")
    folder = os.path.join(root, "ml-20m")
    if not os.path.exists(folder):
        _download(url, zip_path)
        _extract(zip_path, root)

    path = os.path.join(folder, "ratings.csv")
    rng = np.random.RandomState(int(seed))

    try:
        import pandas as pd
        df = pd.read_csv(path)
        if max_ratings is not None and len(df) > max_ratings:
            df = df.sample(n=int(max_ratings), random_state=int(seed))
        user = df["userId"].to_numpy(dtype=np.int64)
        item = df["movieId"].to_numpy(dtype=np.int64)
        rating = df["rating"].to_numpy(dtype=np.float32)
        ts = df["timestamp"].to_numpy(dtype=np.int64)
        return user, item, rating, ts
    except Exception:
        import csv
        user_list, item_list, rating_list, ts_list = [], [], [], []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            _ = next(reader, None)
            for row in reader:
                if not row or len(row) < 4:
                    continue
                user_list.append(int(row[0]))
                item_list.append(int(row[1]))
                rating_list.append(float(row[2]))
                ts_list.append(int(row[3]))
        user = np.asarray(user_list, dtype=np.int64)
        item = np.asarray(item_list, dtype=np.int64)
        rating = np.asarray(rating_list, dtype=np.float32)
        ts = np.asarray(ts_list, dtype=np.int64)

        if max_ratings is not None and len(rating) > max_ratings:
            idx = rng.choice(np.arange(len(rating)), size=int(max_ratings), replace=False)
            user, item, rating, ts = user[idx], item[idx], rating[idx], ts[idx]
        return user, item, rating, ts


def _remap_ids(user_id_raw, item_id_raw):
    uniq_u = np.unique(user_id_raw)
    uniq_i = np.unique(item_id_raw)
    u_map = {u: idx for idx, u in enumerate(uniq_u)}
    i_map = {i: idx for idx, i in enumerate(uniq_i)}
    user = np.array([u_map[u] for u in user_id_raw], dtype=np.int64)
    item = np.array([i_map[i] for i in item_id_raw], dtype=np.int64)
    return user, item, len(uniq_u), len(uniq_i), u_map, i_map


def load_genre_map_ml1m(root):
    path = os.path.join(root, "ml-1m", "movies.dat")
    movieid_to_genres = {}
    genres = set()
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            mid, title, g = line.strip().split("::")
            glist = g.split("|")
            mid = int(mid)
            movieid_to_genres[mid] = glist
            for x in glist:
                genres.add(x)
    genre_list = sorted(list(genres))
    if "unknown" not in genre_list:
        genre_list = ["unknown"] + genre_list
    genre_to_gid = {g: i for i, g in enumerate(genre_list)}
    return movieid_to_genres, genre_to_gid, genre_list


def load_genre_map_ml100k(root):
    path = os.path.join(root, "ml-100k", "u.item")
    genre_names = [
        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    genre_to_gid = {g: i for i, g in enumerate(genre_names)}
    movieid_to_genres = {}

    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            mid = int(parts[0])
            flags = parts[-19:]
            glist = [genre_names[i] for i, v in enumerate(flags) if v == "1"]
            if len(glist) == 0:
                glist = ["unknown"]
            movieid_to_genres[mid] = glist

    return movieid_to_genres, genre_to_gid, genre_names


def load_genre_map_ml20m(root):
    folder = os.path.join(root, "ml-20m")
    path = os.path.join(folder, "movies.csv")

    movieid_to_genres = {}
    genres = set()

    try:
        import pandas as pd
        df = pd.read_csv(path)
        for mid, g in zip(df["movieId"].to_numpy(), df["genres"].to_numpy()):
            mid = int(mid)
            if isinstance(g, str) and g.strip() and g != "(no genres listed)":
                glist = g.split("|")
            else:
                glist = ["unknown"]
            movieid_to_genres[mid] = glist
            for x in glist:
                genres.add(x)
    except Exception:
        import csv
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            _ = next(reader, None)
            for row in reader:
                if not row or len(row) < 3:
                    continue
                mid = int(row[0])
                g = row[-1]
                if g and g != "(no genres listed)":
                    glist = g.split("|")
                else:
                    glist = ["unknown"]
                movieid_to_genres[mid] = glist
                for x in glist:
                    genres.add(x)

    genre_list = sorted(list(genres))
    if "unknown" not in genre_list:
        genre_list = ["unknown"] + genre_list
    genre_to_gid = {g: i for i, g in enumerate(genre_list)}
    return movieid_to_genres, genre_to_gid, genre_list


def build_movieid_seeded_gid(movieid_to_genres, genre_to_gid, seed=0):
    rng = np.random.RandomState(int(seed))
    fallback = "unknown" if "unknown" in genre_to_gid else list(genre_to_gid.keys())[0]
    fallback_gid = int(genre_to_gid[fallback])

    max_mid = 0
    for mid in movieid_to_genres.keys():
        max_mid = max(max_mid, int(mid))

    arr = np.full((max_mid + 1,), fallback_gid, dtype=np.int64)

    for mid, glist in movieid_to_genres.items():
        if (not glist) or (len(glist) == 0):
            gids = np.array([fallback_gid], dtype=np.int64)
        else:
            gids = np.array([int(genre_to_gid.get(g, fallback_gid)) for g in glist], dtype=np.int64)
            gids = np.unique(gids)
        arr[int(mid)] = int(rng.choice(gids))
    return arr


def _force_topk_genres(primary_gid_all, genre_names, top_k=10, source="all"):
    primary_gid_all = np.asarray(primary_gid_all, dtype=np.int64)
    G = len(genre_names)
    cnt = np.bincount(primary_gid_all, minlength=G).astype(np.int64)
    order = np.argsort(-cnt)
    order = [int(g) for g in order if cnt[g] > 0]
    keep_old = order[: int(top_k)]
    keep_old_set = set(keep_old)

    keep_mask = np.array([int(g) in keep_old_set for g in primary_gid_all], dtype=np.bool_)
    old2new = {old: new for new, old in enumerate(keep_old)}
    new_gid_all = np.array([old2new[int(g)] for g in primary_gid_all[keep_mask].tolist()], dtype=np.int64)
    new_genre_names = [genre_names[g] for g in keep_old]

    top_pairs = [(genre_names[g], int(cnt[g])) for g in keep_old]
    print(f"[MovieLens] Force Genres (G={len(new_genre_names)}) by top-{len(new_genre_names)} freq (source={source}).")
    print("[MovieLens] Top genres kept:", ", ".join([f"{n}:{c}" for n, c in top_pairs]))
    return keep_mask, new_gid_all, new_genre_names, old2new, top_pairs


def _cap_selected_genres(new_gid_all, genre_names, head_genres=("Drama", "Comedy"), cap_ratio=1.2, seed=0):
    rng = np.random.RandomState(int(seed))
    new_gid_all = np.asarray(new_gid_all, dtype=np.int64)
    G = len(genre_names)
    before = np.bincount(new_gid_all, minlength=G).astype(np.int64)

    nz = before[before > 0]
    if nz.size == 0:
        keep = np.arange(len(new_gid_all), dtype=np.int64)
        return keep, before, before, 0

    med = int(np.median(nz))
    cap = int(max(1, round(med * float(cap_ratio))))

    head_set = set(head_genres)
    keep_idx_list = []
    for g in range(G):
        idx = np.where(new_gid_all == g)[0]
        if idx.size == 0:
            continue
        name = genre_names[g]
        if name in head_set and idx.size > cap:
            pick = rng.choice(idx, size=cap, replace=False)
            keep_idx_list.append(pick)
        else:
            keep_idx_list.append(idx)

    keep = np.concatenate(keep_idx_list).astype(np.int64)
    keep.sort()
    after = np.bincount(new_gid_all[keep], minlength=G).astype(np.int64)
    return keep, before, after, cap


def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return 0.5 * (kl_pm + kl_qm)


def js_matrix_from_distributions(dists):
    n = dists.shape[0]
    JS = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            JS[i, j] = js_divergence(dists[i], dists[j])
    return JS


def print_client_genre_matrix(counts, dists, genre_names, topk=5):
    C, G = counts.shape
    print("\n================== [MovieLens] Client-Genre Matrix ==================")
    print(f"Genres (G={G}): {genre_names}")
    np.set_printoptions(suppress=True, precision=4)
    print("\n[Counts] shape = (clients, genres) = ", counts.shape)
    print(np.round(counts, 1))

    print("\n[Distributions] (rows sum to 1) shape = ", dists.shape)
    print(np.round(dists, 4))

    print(f"\n[Top-{topk} genres per client]")
    for cid in range(C):
        order = np.argsort(-dists[cid])[:topk]
        pairs = [(genre_names[g], float(dists[cid][g])) for g in order]
        print(f"Client {cid}: " + ", ".join([f"{name}:{val:.3f}" for name, val in pairs]))
    print("=====================================================================\n")


def _sample_dirichlet_props_per_genre(n_genres, n_clients, beta, rng):
    beta = float(beta)
    C = int(n_clients)
    G = int(n_genres)
    alpha = np.full((C,), max(1e-8, beta), dtype=np.float64)
    props = np.zeros((G, C), dtype=np.float64)
    for g in range(G):
        p = rng.dirichlet(alpha)
        props[g] = p
    return props


def _assign_by_given_props(label_arr, props_gc, seed=0):
    rng = np.random.RandomState(int(seed))
    label_arr = np.asarray(label_arr, dtype=np.int64)
    G, C = props_gc.shape
    N = int(label_arr.shape[0])

    assigned = -np.ones((N,), dtype=np.int64)
    for g in range(G):
        idx = np.where(label_arr == g)[0]
        if idx.size == 0:
            continue
        rng.shuffle(idx)
        p = props_gc[g].copy()
        p = np.maximum(p, 0.0)
        s = p.sum()
        if s <= 1e-12:
            p[:] = 1.0 / float(C)
        else:
            p /= s

        counts = rng.multinomial(int(idx.size), pvals=p)
        start = 0
        for c in range(C):
            k = int(counts[c])
            if k <= 0:
                continue
            part = idx[start:start + k]
            assigned[part] = c
            start += k

    miss = np.where(assigned < 0)[0]
    if miss.size > 0:
        assigned[miss] = rng.randint(0, C, size=int(miss.size))
    return assigned


def _dirichlet_partition_interactions_by_genre(
    g_train, n_clients, beta, seed=0,
    min_interactions_per_client=0,
    max_retries=50,
):
    rng = np.random.RandomState(int(seed))
    g_train = np.asarray(g_train, dtype=np.int64)
    C = int(n_clients)
    G = int(g_train.max() + 1) if g_train.size > 0 else 0

    if G <= 0:
        assigned = np.zeros((int(g_train.shape[0]),), dtype=np.int64)
        props = np.full((1, C), 1.0 / float(C), dtype=np.float64)
        return assigned, props

    best = None
    for _ in range(int(max_retries)):
        props = _sample_dirichlet_props_per_genre(G, C, beta, rng)
        assigned = _assign_by_given_props(g_train, props, seed=rng.randint(0, 10**9))
        if int(min_interactions_per_client) <= 0:
            return assigned, props
        cnt = np.bincount(assigned, minlength=C).astype(np.int64)
        if cnt.min() >= int(min_interactions_per_client):
            return assigned, props
        gap = int(cnt.max() - cnt.min())
        if best is None or gap < best[0]:
            best = (gap, assigned, props)

    if best is not None:
        return best[1], best[2]
    props = _sample_dirichlet_props_per_genre(G, C, beta, rng)
    assigned = _assign_by_given_props(g_train, props, seed=rng.randint(0, 10**9))
    return assigned, props


class MovieLensImplicitPosDataset(Dataset):
    def __init__(self, user, item):
        self.user = torch.from_numpy(np.asarray(user, dtype=np.int64)).long()
        self.item = torch.from_numpy(np.asarray(item, dtype=np.int64)).long()

    def __len__(self):
        return int(self.user.shape[0])

    def __getitem__(self, idx):
        return self.user[idx], self.item[idx]


def dataset_read(
    dataset,
    base_path,
    batch_size,
    n_parties,
    partition,
    beta,
    skew_class,
    seed=0,
    val_ratio=0.2,
    test_ratio=0.2,
    temporal_split=True,
    pos_threshold=4,
    print_genre_matrix=True,
    sim_mode="one_minus_js",
    sim_tau=0.4,
    max_ratings=None,

    force_topk_genres=True,
    top_k_genres=10,

    balance_head_genres=True,
    head_genres=("Drama", "Comedy", "Action", "Thriller", "Romance"),
    head_cap_ratio=0.5,

    min_interactions_per_client=200,
    max_retries=50,

    **kwargs,
):

    if dataset == "ml100k":
        user_id_raw, item_id_raw, rating, ts = _load_ratings_ml100k(base_path)
        movieid_to_genres, genre_to_gid, genre_names = load_genre_map_ml100k(base_path)
    elif dataset == "ml1m":
        user_id_raw, item_id_raw, rating, ts = _load_ratings_ml1m(base_path)
        movieid_to_genres, genre_to_gid, genre_names = load_genre_map_ml1m(base_path)
    elif dataset == "ml20m":
        user_id_raw, item_id_raw, rating, ts = _load_ratings_ml20m(base_path, max_ratings=max_ratings, seed=seed)
        movieid_to_genres, genre_to_gid, genre_names = load_genre_map_ml20m(base_path)
    else:
        raise ValueError("dataset must be 'ml100k' or 'ml1m' or 'ml20m'")

    user, item, n_users, n_items, u_map, i_map = _remap_ids(user_id_raw, item_id_raw)

    movieid_primary_gid = build_movieid_seeded_gid(movieid_to_genres, genre_to_gid, seed=int(seed) + 77)

    def _genre_of_raw_movieids(raw_mid_arr):
        raw_mid_arr = np.asarray(raw_mid_arr, dtype=np.int64)
        clipped = np.minimum(raw_mid_arr, movieid_primary_gid.shape[0] - 1)
        return movieid_primary_gid[clipped].astype(np.int64)

    gid_all = _genre_of_raw_movieids(item_id_raw)
    n_genres = len(genre_names)

    if bool(force_topk_genres):
        keep_mask, new_gid_all, new_genre_names, old2new, top_pairs = _force_topk_genres(
            primary_gid_all=gid_all,
            genre_names=genre_names,
            top_k=int(top_k_genres),
            source="all",
        )
        user = user[keep_mask]
        item = item[keep_mask]
        rating = rating[keep_mask]
        ts = ts[keep_mask]
        user_id_raw = user_id_raw[keep_mask]
        item_id_raw = item_id_raw[keep_mask]
        gid_all = new_gid_all
        genre_names = new_genre_names
        n_genres = len(genre_names)

    if bool(balance_head_genres):
        keep_idx, before_cnt, after_cnt, cap_val = _cap_selected_genres(
            new_gid_all=gid_all,
            genre_names=genre_names,
            head_genres=head_genres,
            cap_ratio=float(head_cap_ratio),
            seed=int(seed) + 4242,
        )
        if cap_val > 0:
            def _fmt(cnt):
                pairs = [(genre_names[i], int(cnt[i])) for i in range(len(genre_names))]
                pairs = sorted(pairs, key=lambda x: -x[1])
                return ", ".join([f"{n}:{c}" for n, c in pairs])

            print(f"[MovieLens] Downsample head genres={list(head_genres)} with cap=median*{head_cap_ratio:.3f} (cap={cap_val}).")
            print("[MovieLens] Counts(before):", _fmt(before_cnt))
            print("[MovieLens] Counts(after) :", _fmt(after_cnt))

        user = user[keep_idx]
        item = item[keep_idx]
        rating = rating[keep_idx]
        ts = ts[keep_idx]
        gid_all = gid_all[keep_idx]

    n = int(len(rating))
    idx_all = np.arange(n, dtype=np.int64)
    if bool(temporal_split):
        idx_all = idx_all[np.argsort(ts)]

    n_test = int(float(test_ratio) * n)
    n_test = max(0, min(n_test, n - 1))
    test_idxs = np.asarray(idx_all[-n_test:], dtype=np.int64)
    train_idxs = np.asarray(idx_all[:-n_test], dtype=np.int64)

    train_pos_mask = rating[train_idxs] >= float(pos_threshold)
    test_pos_mask = rating[test_idxs] >= float(pos_threshold)

    train_pos_gidx = train_idxs[train_pos_mask]
    test_pos_gidx = test_idxs[test_pos_mask]

    u_train_pos = user[train_pos_gidx].astype(np.int64)
    i_train_pos = item[train_pos_gidx].astype(np.int64)
    g_train_pos = gid_all[train_pos_gidx].astype(np.int64)

    u_test_pos = user[test_pos_gidx].astype(np.int64)
    i_test_pos = item[test_pos_gidx].astype(np.int64)
    g_test_pos = gid_all[test_pos_gidx].astype(np.int64)

    part = str(partition).lower()
    C = int(n_parties)

    if ("dirichlet" in part) or ("noniid" in part) or ("skew" in part):
        assigned_train, props_gc = _dirichlet_partition_interactions_by_genre(
            g_train=g_train_pos,
            n_clients=C,
            beta=float(beta),
            seed=int(seed) + 2026,
            min_interactions_per_client=int(min_interactions_per_client),
            max_retries=int(max_retries),
        )
        assigned_test = _assign_by_given_props(g_test_pos, props_gc, seed=int(seed) + 9090)
        print(f"[MovieLens] Partition = Dirichlet(interaction-by-genre) | beta={beta}")
    elif ("iid" in part) or ("homo" in part):
        rng = np.random.RandomState(int(seed) + 2027)
        assigned_train = rng.randint(0, C, size=int(g_train_pos.shape[0])).astype(np.int64)
        assigned_test = rng.randint(0, C, size=int(g_test_pos.shape[0])).astype(np.int64)
        props_gc = None
        print(f"[MovieLens] Partition = IID(interaction-random)")
    else:
        assigned_train, props_gc = _dirichlet_partition_interactions_by_genre(
            g_train=g_train_pos,
            n_clients=C,
            beta=float(beta),
            seed=int(seed) + 2028,
            min_interactions_per_client=int(min_interactions_per_client),
            max_retries=int(max_retries),
        )
        assigned_test = _assign_by_given_props(g_test_pos, props_gc, seed=int(seed) + 9091)
        print(f"[MovieLens][WARN] partition='{partition}' not recognized. Fallback to Dirichlet(interaction-by-genre).")

    net_local_map = {cid: np.where(assigned_train == cid)[0].astype(np.int64) for cid in range(C)}
    net_dataidx_map = {cid: train_pos_gidx[net_local_map[cid]].tolist() for cid in range(C)}
    test_local_map = {cid: np.where(assigned_test == cid)[0].astype(np.int64) for cid in range(C)}

    traindata_cls_counts = np.zeros((C, int(n_genres)), dtype=np.float64)
    for cid in range(C):
        local = net_local_map[cid]
        if local.size == 0:
            continue
        traindata_cls_counts[cid] = np.bincount(g_train_pos[local], minlength=int(n_genres)).astype(np.float64)

    row_sum = traindata_cls_counts.sum(axis=1, keepdims=True)
    data_distributions = np.where(
        row_sum > 0,
        traindata_cls_counts / np.maximum(row_sum, 1e-12),
        1.0 / float(n_genres)
    )

    js_div = js_matrix_from_distributions(data_distributions)
    if sim_mode == "exp":
        js_matrix = np.exp(-js_div / max(float(sim_tau), 1e-12))
        np.fill_diagonal(js_matrix, 1.0)
    else:
        js_matrix = 1.0 - js_div
        np.fill_diagonal(js_matrix, 1.0)

    if bool(print_genre_matrix):
        print(f"[MovieLens] pos_threshold={pos_threshold}, C={C}, G={n_genres}")
        print_client_genre_matrix(traindata_cls_counts, data_distributions, genre_names, topk=5)
        print("[MovieLens] 1-JS:", np.round(js_matrix[:20, :20], 8))

    user_train_items = [set() for _ in range(int(n_users))]
    if u_train_pos.size > 0:
        order = np.argsort(u_train_pos)
        u_sorted = u_train_pos[order]
        i_sorted = i_train_pos[order]
        split = np.where(np.diff(u_sorted) != 0)[0] + 1
        u_groups = np.split(u_sorted, split)
        i_groups = np.split(i_sorted, split)
        for ug, ig in zip(u_groups, i_groups):
            if ug.size == 0:
                continue
            uu = int(ug[0])
            user_train_items[uu] = set(np.unique(ig).tolist())

    train_dataloaders, val_dataloaders, test_dataloaders = [], [], []
    client_test_pos_items = [dict() for _ in range(C)]
    client_user_sets = []

    for cid in range(C):
        local = net_local_map[cid]
        rng = np.random.RandomState(int(seed) + 10 + cid)
        local = local.copy()
        rng.shuffle(local)

        n_local = int(local.size)
        if n_local <= 0:
            train_dataloaders.append(None)
            val_dataloaders.append(None)
            test_dataloaders.append(None)
            client_user_sets.append(set())
            print(f"[Client {cid}] train_pos=0 | test_pos={int(test_local_map[cid].size)}")
            continue

        n_val = int(float(val_ratio) * n_local)
        n_val = max(0, min(n_val, n_local - 1))
        val_local = local[:n_val]
        tr_local = local[n_val:]

        tr_gidx = train_pos_gidx[tr_local]
        val_gidx = train_pos_gidx[val_local]

        train_ds = MovieLensImplicitPosDataset(user[tr_gidx], item[tr_gidx])
        val_ds = MovieLensImplicitPosDataset(user[val_gidx], item[val_gidx])

        train_dataloaders.append(DataLoader(train_ds, batch_size=int(batch_size), shuffle=True))
        val_dataloaders.append(DataLoader(val_ds, batch_size=int(batch_size), shuffle=False))

        client_user_sets.append(set(np.unique(user[tr_gidx]).astype(np.int64).tolist()))

        tloc = test_local_map[cid]
        if tloc.size == 0:
            test_dataloaders.append(None)
        else:
            tgidx = test_pos_gidx[tloc]
            test_ds = MovieLensImplicitPosDataset(user[tgidx], item[tgidx])
            test_dataloaders.append(DataLoader(test_ds, batch_size=int(batch_size), shuffle=False))

            uu = user[tgidx].astype(np.int64)
            ii = item[tgidx].astype(np.int64)
            for u0, it0 in zip(uu.tolist(), ii.tolist()):
                d = client_test_pos_items[cid].get(int(u0), None)
                if d is None:
                    client_test_pos_items[cid][int(u0)] = set([int(it0)])
                else:
                    d.add(int(it0))

        print(f"[Client {cid}] train_pos={len(tr_gidx)} | val_pos={len(val_gidx)} | test_pos={int(tloc.size)}")

    data_sizes = [len(net_dataidx_map[cid]) for cid in range(C)]
    user_sizes = [len(client_user_sets[cid]) for cid in range(C)]
    print(
        f"[MovieLens] dataset={dataset} partition={partition} | "
        f"train_pos mean={np.mean(data_sizes):.2f}, std={np.std(data_sizes):.2f} | "
        f"users mean={np.mean(user_sizes):.2f}, std={np.std(user_sizes):.2f}"
    )

    meta = dict(
        n_users=int(n_users),
        n_items=int(n_items),
        genre_names=genre_names,
        user_train_items=user_train_items,
        client_user_sets=client_user_sets,
        client_test_pos_items=client_test_pos_items,
        pos_threshold=float(pos_threshold),
        js_divergence=js_div,
        forced_topk=bool(force_topk_genres),
        top_k_genres=int(top_k_genres),
        head_downsample=bool(balance_head_genres),
        head_genres=list(head_genres),
        head_cap_ratio=float(head_cap_ratio),
        partition=str(partition),
        beta=float(beta),
        dirichlet_props_gc=props_gc,
    )

    return (
        train_dataloaders, val_dataloaders, test_dataloaders,
        net_dataidx_map, traindata_cls_counts, data_distributions,
        js_matrix, meta
    )
