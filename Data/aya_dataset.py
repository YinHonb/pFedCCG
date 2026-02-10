import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, concatenate_datasets

# =========================================================
# 1) Multilingual system prompts
# =========================================================
MULTILINGUAL_SYSTEM_PROMPTS = {
    "plt": "Mpanampy mahasoa ianao.",
    "sin": "ඔබ සහායක සහකරුවෙකි.",
    "tam": "நீங்கள் ஒரு பயனுள்ள உதவியாளர்.",
    "yor": "Olùrànlọ́wọ́ to wúlò ni ọ́.",
    "zsm": "Anda adalah pembantu yang sangat membantu.",
    "por": "Você é um assistente prestativo.",
    "vie": "Bạn là một trợ lý hữu ích.",
    "kir": "Сиз пайдалуу жардамчысыз.",
    "tel": "మీరు సహాయకరమైన సహాయకుడు.",
    "ary": "نتا معاون مزيان.",
    "eng": "You are a helpful assistant.",
    "zho": "你是一个乐于助人的助手。",
    "chi": "你是一个乐于助人的助手。",
    "fra": "Vous êtes un assistant utile.",
    "rus": "Вы — полезный помощник.",
    "spa": "Eres un asistente útil.",
    "ara": "أنت مساعد مفيد.",
    "default": "You are a helpful assistant.",
}

# =========================================================
# 2) Qwen2.5 tokenizer fix
# =========================================================
_QWEN25_CHATML_TEMPLATE = r"""{% for message in messages %}
{% if message['role'] == 'system' %}<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""


def ensure_qwen25_tokenizer(tokenizer, verbose=True):
    changed = False

    if getattr(tokenizer, "chat_template", None) in [None, ""]:
        tokenizer.chat_template = _QWEN25_CHATML_TEMPLATE
        changed = True
        if verbose:
            print("[TokenizerFix] tokenizer.chat_template was None -> set Qwen2.5 ChatML template.")

    try:
        for tok in ["<|end|>", "<|im_end|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0:
                if tokenizer.eos_token != tok:
                    old = tokenizer.eos_token
                    tokenizer.eos_token = tok
                    changed = True
                    if verbose:
                        print(f"[TokenizerFix] eos_token {old} -> {tok} (id={tid}).")
                break
    except Exception:
        pass

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        changed = True
        if verbose:
            print(f"[TokenizerFix] pad_token_id was None -> set pad_token=eos_token ({tokenizer.pad_token}).")

    return changed


# =========================================================
# 3) Dataset / Collator
# =========================================================
class AyaSFTDataset(Dataset):
    def __init__(self, hf_tok_ds):
        self.ds = hf_tok_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        return {
            "input_ids": ex["input_ids"],
            "attention_mask": ex["attention_mask"],
            "labels": ex["labels"],
            "lang_id": ex["lang_id"],
        }


class SFTCollator:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.pad_id = int(self.tok.pad_token_id)

    def __call__(self, batch):
        B = len(batch)
        lengths = [len(b["input_ids"]) for b in batch]
        max_len = max(lengths) if lengths else 1

        input_ids = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)
        lang_id = torch.tensor([b["lang_id"] for b in batch], dtype=torch.long)

        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            input_ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
            attention_mask[i, :L] = 1
            labels[i, :L] = torch.tensor(b["labels"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "lang_id": lang_id,
        }


# =========================================================
# 4) Tokenization helpers
# =========================================================
def _infer_text_fields(ex):
    prompt = None
    resp = None

    for k in ["inputs", "prompt", "instruction", "question", "input", "text"]:
        if k in ex and ex[k] is not None:
            prompt = ex[k]
            break

    for k in ["targets", "response", "output", "answer", "completion"]:
        if k in ex and ex[k] is not None:
            resp = ex[k]
            break

    if prompt is None or resp is None:
        raise KeyError(f"[Aya] Cannot infer prompt/response fields from keys={list(ex.keys())}")

    return str(prompt), str(resp)


def _tokenize_filter_map(
    raw_ds,
    tokenizer,
    max_length,
    lang2id,
    lang_col,
    debug_prefix="[Aya]",
    print_every=2000,
    drop_zero_supervised=True,
):
    max_length = int(max_length)
    eos_id = tokenizer.eos_token_id

    st = {
        "seen": 0,
        "kept": 0,
        "drop_long": 0,
        "drop_prefix": 0,
        "drop_zero_sup": 0,
        "sum_sup": 0,
    }

    kept_list = []

    for i in range(len(raw_ds)):
        ex = raw_ds[i]
        st["seen"] += 1

        prompt, resp = _infer_text_fields(ex)
        lang_code = ex[lang_col]
        if lang_code not in lang2id:
            continue
        lang_id = int(lang2id[lang_code])

        sys_prompt = MULTILINGUAL_SYSTEM_PROMPTS.get(lang_code, MULTILINGUAL_SYSTEM_PROMPTS["default"])

        messages_prompt = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        messages_full = messages_prompt + [{"role": "assistant", "content": resp}]

        prompt_text = tokenizer.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False, truncation=False)["input_ids"]

        if eos_id is not None and (len(full_ids) == 0 or full_ids[-1] != int(eos_id)):
            full_ids = full_ids + [int(eos_id)]

        tok_len = len(full_ids)
        if tok_len > max_length:
            st["drop_long"] += 1
            continue

        if full_ids[:len(prompt_ids)] != prompt_ids:
            st["drop_prefix"] += 1
            continue

        labels = list(full_ids)
        pl = min(len(prompt_ids), len(labels))
        for t in range(pl):
            labels[t] = -100

        n_sup = int(np.sum(np.array(labels) != -100))
        if drop_zero_supervised and n_sup == 0:
            st["drop_zero_sup"] += 1
            continue

        kept_list.append({
            "input_ids": full_ids,
            "attention_mask": [1] * tok_len,
            "labels": labels,
            "lang_id": lang_id,
            lang_col: lang_code,
        })

        st["kept"] += 1
        st["sum_sup"] += n_sup

        if (st["seen"] % print_every) == 0:
            avg_sup = st["sum_sup"] / max(st["kept"], 1)
            print(f"{debug_prefix}[DEBUG] seen={st['seen']} kept={st['kept']} "
                  f"drop_long={st['drop_long']} drop_prefix={st['drop_prefix']} drop_zero_sup={st['drop_zero_sup']} "
                  f"avg_sup(kept)={avg_sup:.2f}")

    tok_ds = HFDataset.from_list(kept_list)

    avg_sup = st["sum_sup"] / max(st["kept"], 1)
    print(f"{debug_prefix}[DEBUG] ===== TOKENIZE SUMMARY =====")
    print(f"{debug_prefix}[DEBUG] total_seen={st['seen']} kept={st['kept']} "
          f"drop_long={st['drop_long']} drop_prefix={st['drop_prefix']} drop_zero_sup={st['drop_zero_sup']}")
    print(f"{debug_prefix}[DEBUG] avg_supervised_tokens(kept)={avg_sup:.2f}")
    print(f"{debug_prefix}[DEBUG] ============================")

    return tok_ds


def _count_per_lang(tok_ds, n_langs):
    if len(tok_ds) == 0:
        return np.zeros((n_langs,), dtype=np.int64)
    y = np.array(tok_ds["lang_id"], dtype=np.int64)
    out = np.zeros((n_langs,), dtype=np.int64)
    for k in range(n_langs):
        out[k] = int(np.sum(y == k))
    return out


# =========================================================
# 5) Non-IID train partition: each client gets exactly K languages
# =========================================================
def _sample_client_langs(n_clients, n_langs, k_langs_per_client, seed=42):
    rng = np.random.default_rng(int(seed))
    k = int(k_langs_per_client)
    k = max(1, min(k, n_langs))

    client_langs = []
    for c in range(n_clients):
        client_langs.append(set(rng.choice(n_langs, size=k, replace=False).tolist()))

    all_langs = set().union(*client_langs) if client_langs else set()
    missing = [l for l in range(n_langs) if l not in all_langs]
    for l in missing:
        c = int(rng.integers(0, n_clients))
        if len(client_langs[c]) >= k:
            drop = rng.choice(list(client_langs[c]))
            client_langs[c].remove(int(drop))
        client_langs[c].add(int(l))

    return [sorted(list(s)) for s in client_langs]


def _partition_random_k_langs(
    labels,
    n_clients,
    n_langs,
    k_langs_per_client=2,
    min_require_size=10,
    seed=42,
    max_tries=50,
    verbose=True,
):
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    N = len(labels)

    if N == 0:
        return {i: [] for i in range(n_clients)}, _sample_client_langs(n_clients, n_langs, k_langs_per_client, seed)

    min_require_size = int(min_require_size)

    last_idx_batch = None
    last_client_lang_ids = None

    for t in range(int(max_tries)):
        client_lang_ids = _sample_client_langs(n_clients, n_langs, k_langs_per_client, seed + t)

        lang2clients = {k: [] for k in range(n_langs)}
        for c in range(n_clients):
            for k in client_lang_ids[c]:
                lang2clients[int(k)].append(c)

        idx_batch = [[] for _ in range(n_clients)]

        for k in range(n_langs):
            idx_k = np.where(labels == k)[0]
            if len(idx_k) == 0:
                continue
            rng.shuffle(idx_k)

            cs = lang2clients[k]
            if len(cs) == 0:
                cs = [int(rng.integers(0, n_clients))]

            splits = np.array_split(idx_k, len(cs))
            for part, c in zip(splits, cs):
                idx_batch[int(c)].extend(part.tolist())

        sizes = [len(x) for x in idx_batch]
        last_idx_batch = idx_batch
        last_client_lang_ids = client_lang_ids

        if min(sizes) >= min_require_size or N < n_clients * min_require_size:
            net_dataidx_map = {}
            for c in range(n_clients):
                rng.shuffle(idx_batch[c])
                net_dataidx_map[c] = idx_batch[c]
            return net_dataidx_map, client_lang_ids

    if verbose:
        print("[Aya][WARN] random-K partition couldn't satisfy min_require_size; using last try.")
    net_dataidx_map = {c: last_idx_batch[c] for c in range(n_clients)}
    return net_dataidx_map, last_client_lang_ids


def _kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.where(p <= 0, 1e-12, p)
    q = np.where(q <= 0, 1e-12, q)
    return float(np.sum(p * np.log2(p / q)))


def _js_similarity(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    return float(1.0 - 0.5 * (_kl_divergence(p, m) + _kl_divergence(q, m)))


def _calculate_js_matrix(dist_mat):
    n = dist_mat.shape[0]
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            out[i, j] = _js_similarity(dist_mat[i], dist_mat[j])
    return out


def _record_net_data_stats(y, net_dataidx_map, n_classes, verbose=True):
    y = np.asarray(y, dtype=np.int64)
    n_clients = len(net_dataidx_map)

    counts = np.zeros((n_clients, n_classes), dtype=np.int64)
    sizes = np.zeros((n_clients,), dtype=np.int64)

    for c in range(n_clients):
        idx = np.asarray(net_dataidx_map[c], dtype=np.int64)
        sizes[c] = len(idx)
        if len(idx) == 0:
            continue
        yy = y[idx]
        unq, unq_cnt = np.unique(yy, return_counts=True)
        for k, cnt in zip(unq, unq_cnt):
            counts[c, int(k)] = int(cnt)

    dist = counts.astype(np.float64)
    dist = dist / (dist.sum(axis=1, keepdims=True) + 1e-12)
    js_matrix = _calculate_js_matrix(dist)

    if verbose:
        print(f"[Aya] client train size: mean={sizes.mean():.2f}, std={sizes.std():.2f}, min={sizes.min()}, max={sizes.max()}")
        print("[Aya] Train lang-counts per client (rows=clients, cols=langs):")
        print(counts)
        print("[Aya] JS-matrix stats (1-JS similarity): "
              f"min={js_matrix.min():.4f}, mean={js_matrix.mean():.4f}, max={js_matrix.max():.4f}")
        print("[Aya] JS-matrix stats (1-JS similarity): ", js_matrix)

    return counts, dist, js_matrix


def dataset_read_aya(
    tokenizer,
    batch_size: int,
    n_parties: int,
    top_k_langs: int,
    max_length: int = 512,
    seed: int = 42,
    verbose: bool = True,

    train_per_lang: int = 1000,
    test_per_lang: int = 200,
    oversample_factor: float = 3.0,

    k_langs_per_client: int = 2,
    min_require_size: int = 10,

    num_workers: int = 0,
):

    ensure_qwen25_tokenizer(tokenizer, verbose=verbose)

    lang_col = "language_code"
    dataset_name = "CohereForAI/aya_dataset"

    if verbose:
        print(f"[Aya] Loading split=train only (self-construct test): {dataset_name}")
    try:
        raw = load_dataset(dataset_name, split="train")
    except Exception as e:
        if verbose:
            print(f"[Aya] Failed to load {dataset_name}, fallback to CohereLabs/aya_dataset. err={e}")
        raw = load_dataset("CohereLabs/aya_dataset", split="train")

    if lang_col not in raw.features:
        raise KeyError(f"[Aya] column '{lang_col}' not found in dataset features")

    if verbose:
        print(f"[Aya] Raw train size = {len(raw)}")

    langs = np.array(raw[lang_col])
    uniq, cnt = np.unique(langs, return_counts=True)
    order = np.argsort(-cnt)

    top_k_langs = int(top_k_langs)
    picked = [(str(uniq[i]), int(cnt[i])) for i in order[:top_k_langs]]
    selected_lang_codes = [x[0] for x in picked]
    n_langs = len(selected_lang_codes)

    if verbose:
        print(f"[Aya] Selected top-{n_langs} languages by TRAIN frequency:")
        for lc, c in picked:
            print(f"  - {lc}: {c}")

    lang2id = {lc: i for i, lc in enumerate(selected_lang_codes)}

    rng = np.random.default_rng(int(seed))
    train_list = []
    test_list = []

    target_total = int(train_per_lang) + int(test_per_lang)
    over = max(1, int(np.ceil(float(target_total) * float(oversample_factor))))

    if verbose:
        print(f"[Aya] Target after-tokenize counts: train_per_lang={train_per_lang}, test_per_lang={test_per_lang}")
        print(f"[Aya] Raw oversample per lang: {over} (oversample_factor={oversample_factor})")

    for lc in selected_lang_codes:
        idx = np.where(langs == lc)[0]
        rng.shuffle(idx)

        take = min(len(idx), over)
        sub = raw.select(idx[:take].tolist())

        if verbose:
            print(f"[Aya] ---- Language {lc}: raw_available={len(idx)}, raw_take={take} ----")

        tok_l = _tokenize_filter_map(
            sub,
            tokenizer=tokenizer,
            max_length=max_length,
            lang2id=lang2id,
            lang_col=lang_col,
            debug_prefix=f"[Aya][Tok-{lc}]",
            print_every=2000,
            drop_zero_supervised=True,
        )

        perm = rng.permutation(len(tok_l)) if len(tok_l) > 0 else np.array([], dtype=np.int64)
        tok_l = tok_l.select(perm.tolist()) if len(perm) > 0 else tok_l

        avail = len(tok_l)
        want_te = int(test_per_lang)
        want_tr = int(train_per_lang)

        te_k = min(want_te, avail)
        tr_k = min(want_tr, max(0, avail - te_k))

        if verbose and (te_k < want_te or tr_k < want_tr):
            print(f"[Aya][WARN] {lc}: insufficient after tokenize. avail={avail}, "
                  f"use test={te_k}/{want_te}, train={tr_k}/{want_tr}. "
                  f"(Try larger max_length or oversample_factor)")

        tok_te = tok_l.select(list(range(0, te_k))) if te_k > 0 else HFDataset.from_list([])
        tok_tr = tok_l.select(list(range(te_k, te_k + tr_k))) if tr_k > 0 else HFDataset.from_list([])

        test_list.append(tok_te)
        train_list.append(tok_tr)

    tok_train_all = concatenate_datasets(train_list) if len(train_list) > 0 else HFDataset.from_list([])
    tok_test_all = concatenate_datasets(test_list) if len(test_list) > 0 else HFDataset.from_list([])

    # final shuffle
    if len(tok_train_all) > 0:
        tok_train_all = tok_train_all.select(rng.permutation(len(tok_train_all)).tolist())
    if len(tok_test_all) > 0:
        tok_test_all = tok_test_all.select(rng.permutation(len(tok_test_all)).tolist())

    if verbose:
        tr_cnt = _count_per_lang(tok_train_all, n_langs)
        te_cnt = _count_per_lang(tok_test_all, n_langs)
        print("[Aya] ===== FINAL TOKENIZED COUNTS =====")
        for k, lc in enumerate(selected_lang_codes):
            print(f"  - {k:2d} {lc}: train_tok={int(tr_cnt[k])}, test_tok={int(te_cnt[k])}")
        print(f"[Aya] Total tok_train={len(tok_train_all)}, tok_test={len(tok_test_all)}")
        print("[Aya] ==================================")

    y_train = np.array(tok_train_all["lang_id"], dtype=np.int64)

    net_dataidx_map, client_lang_ids = _partition_random_k_langs(
        labels=y_train,
        n_clients=int(n_parties),
        n_langs=int(n_langs),
        k_langs_per_client=int(k_langs_per_client),
        min_require_size=int(min_require_size),
        seed=int(seed) + 77,
        max_tries=50,
        verbose=verbose,
    )

    lang_counts, data_distributions, js_matrix = _record_net_data_stats(
        y_train, net_dataidx_map, n_classes=int(n_langs), verbose=verbose
    )

    n_clients = int(n_parties)
    client_lang_weights = np.zeros((n_clients, int(n_langs)), dtype=np.float64)
    for c in range(n_clients):
        row = lang_counts[c].astype(np.float64)  # counts per lang for client c
        s = float(row.sum())
        if s > 0:
            client_lang_weights[c] = row / (s + 1e-12)

    if verbose:
        print("[Aya] ===== FINAL TRAIN PARTITION (random K langs) =====")
        for c in range(int(n_parties)):
            langs_c = client_lang_ids[c]
            lang_codes = [selected_lang_codes[k] for k in langs_c]
            tr_size = len(net_dataidx_map.get(c, []))

            row = lang_counts[c]
            nonzero = [(selected_lang_codes[k], int(row[k])) for k in range(n_langs) if int(row[k]) > 0]
            nz_str = ", ".join([f"{lc2}:{cnt2}" for lc2, cnt2 in nonzero]) if nonzero else "(empty)"

            w = client_lang_weights[c]
            w_nz = [(selected_lang_codes[k], float(w[k])) for k in range(n_langs) if w[k] > 0]
            w_str = ", ".join([f"{lc2}:{wk:.3f}" for lc2, wk in w_nz]) if w_nz else "(empty)"

            print(f"  - Client {c:2d}: K={len(langs_c)} langs={lang_codes} | train_size={tr_size}")
            print(f"      train_lang_counts: {nz_str}")
            print(f"      lang_weights(prop): {w_str}")
        print("[Aya] ================================================")

    collator = SFTCollator(tokenizer)
    empty_ds = HFDataset.from_list([])

    train_loaders, val_loaders = [], []
    for c in range(int(n_parties)):
        tr_idx = net_dataidx_map.get(c, [])
        ds_tr = AyaSFTDataset(tok_train_all.select(tr_idx)) if len(tr_idx) > 0 else AyaSFTDataset(empty_ds)
        ds_va = AyaSFTDataset(empty_ds)

        train_loaders.append(DataLoader(
            ds_tr, batch_size=int(batch_size), shuffle=True,
            collate_fn=collator, num_workers=int(num_workers)
        ))
        val_loaders.append(DataLoader(
            ds_va, batch_size=int(batch_size), shuffle=False,
            collate_fn=collator, num_workers=int(num_workers)
        ))

    global_test_loader = DataLoader(
        AyaSFTDataset(tok_test_all),
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=collator,
        num_workers=int(num_workers),
    )
    test_loaders = [global_test_loader for _ in range(int(n_parties))]

    meta = {
        "selected_lang_codes": selected_lang_codes,
        "lang2id": lang2id,
        "n_langs": int(n_langs),
        "max_length": int(max_length),
        "train_per_lang": int(train_per_lang),
        "test_per_lang": int(test_per_lang),
        "oversample_factor": float(oversample_factor),
        "k_langs_per_client": int(k_langs_per_client),
        "min_require_size": int(min_require_size),
        "client_lang_ids": client_lang_ids,
        "client_lang_weights": client_lang_weights.astype(np.float32),
        "note": "Self-construct train/test from train split only; fixed per-lang counts AFTER tokenization; train noniid K langs; global test balanced by construction; personalize by client_lang_weights proportional to TRAIN lang counts.",
    }

    return (
        train_loaders, val_loaders, test_loaders, global_test_loader,
        net_dataidx_map, lang_counts, data_distributions, js_matrix, meta,
    )
