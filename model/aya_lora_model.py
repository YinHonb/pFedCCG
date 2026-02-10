import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


def _guess_lora_targets(model) -> list:
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    found = set()
    for name, _ in model.named_modules():
        last = name.split(".")[-1]
        if last in candidates:
            found.add(last)
    attn = [x for x in ["q_proj", "k_proj", "v_proj", "o_proj"] if x in found]
    if len(attn) >= 2:
        return attn
    return sorted(found) if found else ["q_proj", "v_proj"]


def build_lora_model(
    model_name_or_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict = "auto",
    trust_remote_code: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list | None = None,
):

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    if target_modules is None:
        target_modules = _guess_lora_targets(base)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    return tokenizer, model
