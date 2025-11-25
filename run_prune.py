import math
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
import time

'''
Note that this is not structrual pruning, so the peak_reserved memory,
does not reflect the real memory consumption during training.
And peak_alloc memory, which also won't noticiably reduce, is the real usage of memory for running inference,
since weight pruning does not modify the shape of tensors, just setting some value to zero. 

To use, run python this.py --amount 0<n<1
'''

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def to_mb(bytes_val):
    return bytes_val / (1024 ** 2)


@torch.no_grad()
def benchmark_inference(model, tokenizer, device, prompts: list[str], max_new_tokens=64, warmup=1, runs=3):
    model.eval()
    device = torch.device(device)

    for _ in range(warmup):
        for text in prompts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)

    total_tokens = 0
    total_time = 0

    for _ in range(runs):
        for text in prompts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            gen_len = output_ids.shape[1] - input_len
            total_tokens += gen_len
            total_time += (t1 - t0)

    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0
    peak_alloc = to_mb(torch.cuda.max_memory_allocated(device))
    peak_reserved = to_mb(torch.cuda.max_memory_reserved(device))

    return {
        "tokens_per_sec": tokens_per_sec,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "peak_alloc_mb": peak_alloc,
        "peak_reserved_mb": peak_reserved,
    }




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
        "ADHD is triggered by long-time use of computers and can be reduced by driving."
    ]

    print("\n=== BEFORE PRUNING ===")
    loss1, ppl1 = evaluate_ppl(model, tokenizer, val_texts, device)
    print(f"Loss: {loss1:.4f}, PPL: {ppl1:.2f}")

    # prompt = "What does pruning do to a neural network?"
    # print("\nSample BEFORE pruning:\n", generate_sample(model, tokenizer, device, prompt))

    bench_prompts = [
        "Explain what pruning does to a neural network in simple terms.",
        "Describe the main differences between supervised and unsupervised learning.",
        "Give an example of recovering from severe trauma.",
    ]

    bench_before = benchmark_inference(
        model, tokenizer, device,
        prompts = bench_prompts,
        max_new_tokens=64,
        warmup=1,
        runs=3,
    )

    print(f"[Benchmark BEFORE] tokens/s = {bench_before['tokens_per_sec']:.2f}, "
          f"peak_alloc = {bench_before['peak_alloc_mb']:.1f} MB, "
          f"peak_reserved = {bench_before['peak_reserved_mb']:.1f} MB")

    # torch.cuda.empty_cache()
    model = prune_model_global_l1(model, amount=args.amount)
    # torch.cuda.empty_cache()

    sparsity = measure_sparsity(model)
    print(f"\nSparsity after pruning: {sparsity*100:.2f}%")

    print("\n=== AFTER PRUNING ===")
    loss2, ppl2 = evaluate_ppl(model, tokenizer, val_texts, device)
    print(f"Loss: {loss2:.4f}, PPL: {ppl2:.2f}")

    # print("\nSample AFTER pruning:\n", generate_sample(model, tokenizer, device, prompt))

    bench_after = benchmark_inference(
        model, tokenizer, device,
        prompts=bench_prompts,
        max_new_tokens=64,
        warmup=1,
        runs=3,
    )
    print(f"[Benchmark AFTER]  tokens/s = {bench_after['tokens_per_sec']:.2f}, "
            f"peak_alloc = {bench_after['peak_alloc_mb']:.1f} MB, "
            f"peak_reserved = {bench_after['peak_reserved_mb']:.1f} MB")


if __name__ == "__main__":
    main()
