import numpy as np
import torch
import torch.nn.functional as F
import math

@torch.no_grad()
def evaluate_teacher_forcing_metrics(net, dataloader, device="cuda", topk_list=(1, 5)):
    net.eval()

    total_nll_sum = 0.0
    total_valid_tokens = 0.0
    total_nonpad_tokens = 0.0
    total_ex = 0.0
    zero_valid_ex = 0.0
    correct = {k: 0.0 for k in topk_list}

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = net(input_ids=input_ids, attention_mask=attn)
        logits = out.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_attn = attn[:, 1:].contiguous()

        B, Tm1, V = shift_logits.shape
        total_ex += float(B)

        mask = (shift_labels != -100)
        nonpad = (shift_attn != 0)

        valid_cnt = float(mask.sum().item())
        nonpad_cnt = float(nonpad.sum().item())

        total_valid_tokens += valid_cnt
        total_nonpad_tokens += nonpad_cnt

        if valid_cnt <= 0:
            zero_valid_ex += float(B)
            continue

        loss_flat = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(B, Tm1)

        total_nll_sum += float(loss_flat[mask].sum().item())

        if 1 in topk_list:
            pred = shift_logits.argmax(dim=-1)
            correct[1] += float((pred[mask] == shift_labels[mask]).float().sum().item())

        for k in topk_list:
            if k == 1:
                continue
            top = shift_logits.topk(k=k, dim=-1).indices
            gold = shift_labels.unsqueeze(-1)
            hit = (top == gold).any(dim=-1)
            correct[k] += float(hit[mask].float().sum().item())

    nll = float(total_nll_sum / max(total_valid_tokens, 1.0))
    ppl = float(np.exp(nll))
    bpt = float(nll / math.log(2.0))

    valid_coverage = float(total_valid_tokens / max(total_nonpad_tokens, 1.0))
    avg_valid = float(total_valid_tokens / max(total_ex, 1.0))
    avg_nonpad = float(total_nonpad_tokens / max(total_ex, 1.0))
    zero_valid_frac = float(zero_valid_ex / max(total_ex, 1.0))

    out = {
        "ppl": ppl,
        "nll": nll,
        "bpt": bpt,
        "avg_valid_tokens_per_ex": avg_valid,
        "avg_nonpad_tokens_per_ex": avg_nonpad,
        "valid_coverage": valid_coverage,
        "zero_valid_frac": zero_valid_frac,
        "total_nll_sum": float(total_nll_sum),
        "total_valid_tokens": float(total_valid_tokens),
    }
    for k in topk_list:
        out[f"acc@{k}"] = float(correct[k] / max(total_valid_tokens, 1.0))
    return out

@torch.no_grad()
def evaluate_teacher_forcing_by_lang(
    model,
    dataloader,
    n_langs: int,
    device,
    topk: int = 5,
    lang_key_candidates=("lang_id", "lang", "language_id"),
):
    model.eval()

    L = int(n_langs)
    loss_sum = np.zeros(L, dtype=np.float64)
    loss_sum_sq = np.zeros(L, dtype=np.float64)
    tok_sum = np.zeros(L, dtype=np.float64)
    correct1_sum = np.zeros(L, dtype=np.float64)
    correctk_sum = np.zeros(L, dtype=np.float64)

    def _get_lang_ids(batch):
        for k in lang_key_candidates:
            if k in batch:
                return batch[k]
        return None

    for batch in dataloader:
        lang_ids = _get_lang_ids(batch)
        if lang_ids is None:
            raise KeyError(
                f"Cannot find language id in batch. Tried keys={lang_key_candidates}. "
                f"Batch keys={list(batch.keys())}"
            )

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = batch["labels"].to(device)

        lang_ids = lang_ids.detach().long().view(-1).cpu().numpy()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits  # [B, T, V]

        shifted_logits = logits[:, :-1, :].float()
        shifted_labels = labels[:, 1:]
        mask = (shifted_labels != -100)

        B, Tm1, V = shifted_logits.shape
        k = int(min(max(int(topk), 1), V))

        loss_flat = F.cross_entropy(
            shifted_logits.reshape(-1, V),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        loss_tok = loss_flat.view(B, Tm1)

        pred1 = shifted_logits.argmax(dim=-1)
        correct1 = (pred1 == shifted_labels) & mask


        topk_idx = torch.topk(shifted_logits, k=k, dim=-1).indices
        hitk = (topk_idx == shifted_labels.unsqueeze(-1)) & mask.unsqueeze(-1)
        correctk = hitk.any(dim=-1) & mask

        for b in range(B):
            l = int(lang_ids[b])
            if l < 0 or l >= L:
                continue

            m = mask[b]
            ntok = int(m.sum().item())
            if ntok <= 0:
                continue

            lt = loss_tok[b][m]
            loss_sum[l] += float(lt.sum().item())
            loss_sum_sq[l] += float((lt * lt).sum().item())
            tok_sum[l] += float(ntok)

            correct1_sum[l] += float(correct1[b][m].sum().item())
            correctk_sum[l] += float(correctk[b][m].sum().item())

    per_lang_nll = (loss_sum / (tok_sum + 1e-12)).astype(np.float64)
    per_lang_acc = (correct1_sum / (tok_sum + 1e-12)).astype(np.float64)
    per_lang_topk_acc = (correctk_sum / (tok_sum + 1e-12)).astype(np.float64)

    return {
        "per_lang_nll": per_lang_nll.tolist(),
        "per_lang_tok": tok_sum.tolist(),
        "per_lang_loss_sum": loss_sum.tolist(),
        "per_lang_loss_sum_sq": loss_sum_sq.tolist(),
        "per_lang_acc": per_lang_acc.tolist(),
        "per_lang_top5_acc": per_lang_topk_acc.tolist(),
        "topk": int(k),
    }


@torch.no_grad()
def compute_ppl(net, dataloader, device="cuda"):
    m = evaluate_teacher_forcing_metrics(net, dataloader, device=device, topk_list=())
    return float(m["ppl"])

@torch.no_grad()
def compute_token_accuracy(net, dataloader, device="cuda", topk=1):
    m = evaluate_teacher_forcing_metrics(net, dataloader, device=device, topk_list=(topk,))
    return float(m[f"acc@{topk}"])
