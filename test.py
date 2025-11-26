from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-v1", split="test")
print(dataset)