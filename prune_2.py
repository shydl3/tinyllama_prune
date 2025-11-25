import math
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda:0")
    dtype = torch.float16  # GPU-friendly且省显存

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
    )

    model.to(device)
    print(torch.cuda.get_device_name(0))
    return model, tokenizer, device


@torch.no_grad()
def evaluate_ppl(model, tokenizer, texts, device, max_length=256):
    model.eval()
    losses = []
    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        outputs = model(**enc, labels=enc["input_ids"])
        loss = outputs.loss.detach().float().cpu()
        losses.append(loss)
    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss)
    return float(mean_loss), float(ppl)


@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt, max_new_tokens=80):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    gen = output_ids[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def prune_model_global_l1(model, amount=0.3):
    modules_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            modules_to_prune.append((module, "weight"))

    prune.global_unstructured(
        modules_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    for module, _ in modules_to_prune:
        prune.remove(module, "weight")

    return model



def prune_model_layerwise_l1(model, amount=0.3):
    """
    对每一个 Linear 层单独做 L1 非结构化剪枝。
    不再做全局 topk，避免在 8G 卡上爆显存。
    amount=0.3 表示每层剪掉 30% 最小的权重。
    """
    print(f"\n[Prune] Layer-wise L1 pruning, amount = {amount:.2f}")

    modules_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            modules_to_prune.append((module, "weight"))

    # 对每一层单独剪
    for module, param_name in modules_to_prune:
        prune.l1_unstructured(
            module,
            name=param_name,
            amount=amount,
        )

    # 把 mask merge 回权重里，去掉 reparam
    for module, param_name in modules_to_prune:
        prune.remove(module, param_name)

    print("[Prune] Done.")
    return model


def measure_sparsity(model):
    total, zeros = 0, 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data
            total += W.numel()
            zeros += (W == 0).sum().item()
    return zeros / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amount", type=float, default=0.3)
    args = parser.parse_args()


    model, tokenizer, device = load_model_and_tokenizer()

    val_texts = [
        "The patient was diagnosed with type 2 diabetes.",
        "Sorting algorithms include quicksort and mergesort.",
        "Large language models can be compressed using pruning.",
    ]

    print("\n=== BEFORE PRUNING ===")
    loss1, ppl1 = evaluate_ppl(model, tokenizer, val_texts, device)
    print(f"Loss: {loss1:.4f}, PPL: {ppl1:.2f}")

    prompt = "What does pruning do to a neural network?"
    print("\nSample BEFORE pruning:\n", generate_sample(model, tokenizer, device, prompt))

    # torch.cuda.empty_cache()
    model = prune_model_global_l1(model, amount=args.amount)
    # torch.cuda.empty_cache()

    sparsity = measure_sparsity(model)
    print(f"\nSparsity after pruning: {sparsity*100:.2f}%")

    print("\n=== AFTER PRUNING ===")
    loss2, ppl2 = evaluate_ppl(model, tokenizer, val_texts, device)
    print(f"Loss: {loss2:.4f}, PPL: {ppl2:.2f}")

    print("\nSample AFTER pruning:\n", generate_sample(model, tokenizer, device, prompt))


if __name__ == "__main__":
    main()
