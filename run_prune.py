import math
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
import time

from datasets import load_dataset
import math

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

    device = torch.device("cuda:1")
    dtype = torch.float16  # GPU-friendly且省显存

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # dtype=dtype,
    )

    model.to(device)
    print(f"[DEVICE] {torch.cuda.get_device_name(0)}")
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


def prune_tinyllama_ffn_structured(model, amount: float = 0.2):
    """
    对 TinyLlama 的 FFN 做结构化通道剪枝（每层剪掉 amount 比例的中间维度）。
    - amount=0.2 表示每层删掉 20% 最不重要的 FFN neurons。
    - 重要性度量：down_proj.weight 的列 L1 范数（每个列对应一个中间通道）。
    注意：
      * 在 CPU 上算重要性和阈值，避免 GPU OOM。
      * 直接修改 Linear 的 weight 形状（结构化），不会用 torch.nn.utils.prune。
    """
    assert 0.0 <= amount < 1.0
    print(f"\n[Prune][FFN structured] amount={amount:.2f}")

    # 大多数 LLaMA/TinyLlama 在 HF 里是 model.model.layers
    layers = model.model.layers

    for layer_idx, layer in enumerate(layers):
        mlp = layer.mlp
        up = mlp.up_proj      # [d_ff, d_model]
        gate = mlp.gate_proj  # [d_ff, d_model]
        down = mlp.down_proj  # [d_model, d_ff]

        W_down = down.weight.data  # shape [d_model, d_ff]
        d_model, d_ff = W_down.shape

        if d_ff == 0:
            continue

        # 要剪掉多少个通道
        k_prune = int(amount * d_ff)
        if k_prune < 1:
            continue

        # 1. 在 CPU 上计算每个通道的重要性（这一列的 L1 norm）
        # importance: [d_ff]
        importance = W_down.detach().abs().float().sum(dim=0).cpu()

        # 找到要剪掉的通道阈值
        # kthvalue 第 k 小的值作为阈值，比 topk 更省内存
        threshold, _ = torch.kthvalue(importance, k_prune)

        # 需要保留的通道索引
        keep_mask = importance > threshold
        keep_idx = keep_mask.nonzero(as_tuple=False).view(-1)

        # 确保排序
        keep_idx, _ = torch.sort(keep_idx)

        num_keep = keep_idx.numel()
        pruned_ratio = 1.0 - float(num_keep) / float(d_ff)

        print(f"[Layer {layer_idx}] d_ff={d_ff} -> {num_keep} (pruned {pruned_ratio*100:.1f}%)")

        # 2. 基于 keep_idx 重新构造 FFN 权重（在原设备上）
        device = W_down.device

        # gate_proj & up_proj: 删掉行（输出通道）
        new_gate_weight = gate.weight.data[keep_idx, :].clone().to(device)
        new_up_weight = up.weight.data[keep_idx, :].clone().to(device)
        # down_proj: 删掉列（输入通道）
        new_down_weight = down.weight.data[:, keep_idx].clone().to(device)

        # 替换参数
        gate.out_features = num_keep
        up.out_features = num_keep
        down.in_features = num_keep

        gate.weight = nn.Parameter(new_gate_weight.contiguous())
        up.weight = nn.Parameter(new_up_weight.contiguous())
        down.weight = nn.Parameter(new_down_weight.contiguous())

        # 如果这些线性层有 bias（LLaMA 通常没有），也要同步截取
        if gate.bias is not None:
            gate.bias = nn.Parameter(gate.bias.data[keep_idx].clone().to(device))
        if up.bias is not None:
            up.bias = nn.Parameter(up.bias.data[keep_idx].clone().to(device))
        # down_proj 一般没有 bias，按需处理

    print("[Prune][FFN structured] done.")
    return model

def measure_sparsity(model):
    total, zeros = 0, 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data
            total += W.numel()
            zeros += (W == 0).sum().item()
    return zeros / total


def load_corpus_texts(path, max_texts=2000, max_len=256):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.append(line)
            if len(texts) >= max_texts:
                break
    return texts



@torch.no_grad()
def eval_ppl_on_dataset(model, tokenizer, dataset, device, max_samples=500, max_length=256):
    model.eval()
    losses = []
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break
        text = item["text"]
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

def load_wikitext2_test_tokens(tokenizer, max_chars: int | None = None):
    """
    加载 WikiText-2-v1 的 test split，并用当前 tokenizer 编码成一个长 token 序列。
    返回：torch.LongTensor, shape [num_tokens]
    """
    print("\n[Eval] Loading WikiText-2-v1 test split ...")
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="test")

    # 把所有 text 拼成一个长字符串（标准 LM 评估做法）
    if max_chars is not None:
        texts = []
        total = 0
        for t in dataset["text"]:
            if not t:
                continue
            if total + len(t) > max_chars:
                break
            texts.append(t)
            total += len(t)
        full_text = "\n".join(texts)
    else:
        full_text = "\n".join(dataset["text"])

    enc = tokenizer(full_text, return_tensors="pt")
    input_ids = enc["input_ids"][0]  # shape: [num_tokens]
    print(f"[Eval] WikiText-2 test tokens: {input_ids.shape[0]}")
    return input_ids


@torch.no_grad()
def evaluate_wikitext2_ppl(model, input_ids, device, stride: int = 1024):
    """
    在 WikiText-2 test tokens 上计算 PPL（标准 sliding window 评估）。
    input_ids: LongTensor, shape [T]
    stride: 每个窗口长度（不要超过模型 max_seq_len）
    """
    model.eval()
    device = torch.device(device)
    input_ids = input_ids.to(device)

    nll_sum = 0.0
    total_tokens = 0

    seq_len = input_ids.shape[0]
    # 按 stride 滑动窗口，每个窗口用 labels=input_ids 计算 CE loss
    for start in range(0, seq_len, stride):
        end = min(start + stride, seq_len)
        if end - start < 2:  # 太短没意义
            break
        window = input_ids[start:end].unsqueeze(0)  # [1, L]

        outputs = model(window, labels=window)
        loss = outputs.loss  # 标量 loss（平均每个 token 的负对数似然）

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[Eval][WikiText-2] got NaN/Inf loss at window [{start}:{end}], skip.")
            continue

        n_tokens = end - start
        nll_sum += loss.item() * n_tokens
        total_tokens += n_tokens

    if total_tokens == 0:
        print("[Eval][WikiText-2] No valid tokens, returning NaN.")
        return float("nan"), float("nan")

    mean_nll = nll_sum / total_tokens
    ppl = math.exp(mean_nll)
    return mean_nll, ppl


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

    # val_texts = load_corpus_texts("val_corpus_en.txt", max_texts=1000)

    wikitext_ids = load_wikitext2_test_tokens(
        tokenizer,
        max_chars=None,   # 或者给个上限，比如 200_000，避免太长
    )

    print("\n=== BEFORE PRUNING ===")
    loss1, ppl1 = evaluate_ppl(model, tokenizer, val_texts, device)
    print(f"Loss: {loss1:.4f}, PPL: {ppl1:.2f}")

    loss_wt_before, ppl_wt_before = evaluate_wikitext2_ppl(
        model, wikitext_ids, device=device, stride=1024,
    )
    print(f"[WikiText-2 BEFORE] Loss: {loss_wt_before:.4f}, PPL: {ppl_wt_before:.2f}")


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
    # model = prune_tinyllama_ffn_structured(model, amount=args.amount)
    model = prune_model_global_l1(model, amount=args.amount)
    # torch.cuda.empty_cache()

    sparsity = measure_sparsity(model)
    print(f"\nSparsity after pruning: {sparsity*100:.2f}%")

    print("\n=== AFTER PRUNING ===")
    loss2, ppl2 = eval_ppl_on_dataset(model, tokenizer, raw_dset, device, max_samples=500)
    print(f"Loss: {loss2:.4f}, PPL: {ppl2:.2f}")

    loss_wt_after, ppl_wt_after = evaluate_wikitext2_ppl(
        model, wikitext_ids, device=device, stride=1024,
    )
    print(f"[WikiText-2 AFTER]  Loss: {loss_wt_after:.4f}, PPL: {ppl_wt_after:.2f}")

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
