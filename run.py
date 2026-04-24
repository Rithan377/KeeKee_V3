"""
run.py — End-to-end Demo
==========================
Shows the full pipeline:
  build tokenizer → build model → route prompt → generate → decode

Run:
    python run.py
"""

import torch
from config import GPTConfig, GEMMA3_ATTENTION_PATTERN, AttentionType
from model import KeeKee
from tokenizer import CharTokenizer, SpecialTokens
from router import TaskRouter
from sampler import Sampler

# ─────────────────────────────────────────────
# 1. Build tokenizer from sample corpus
# ─────────────────────────────────────────────
SAMPLE_CORPUS = (
    "The quick brown fox jumps over the lazy dog. "
    "How do I implement a binary search in Python? "
    "Write a short story about a robot who learns to feel. "
    "What is the best route for our shipment from Port A to Warehouse B? "
    "Summarise the key points of the latest supply chain report."
)

tokenizer = CharTokenizer.build_from_text(SAMPLE_CORPUS)
print(f"Vocab size: {tokenizer.vocab_size}")

# ─────────────────────────────────────────────
# 2. Build model (tiny config for demo)
# ─────────────────────────────────────────────
config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    n_layer=6,
    n_embd=128,
    n_head=4,
    n_kv_head=2,               # GQA: 2 kv heads for 4 query heads
    block_size=256,
    attention_pattern=GEMMA3_ATTENTION_PATTERN,   # 5 local + 1 global, repeating
    sliding_window_size=32,
    use_post_attn_norm=True,
    use_post_ffw_norm=True,
    use_qk_norm=True,
    attn_logits_soft_cap=50.0,
    dropout=0.0,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = KeeKee(config).to(device)

n = model.num_params()
print(f"KeeKee v3 | {n:,} params | device={device}")
print(f"Attention types: {[t.value for t in config.attention_types]}")

# ─────────────────────────────────────────────
# 3. Route a prompt → GenerationProfile
# ─────────────────────────────────────────────
router = TaskRouter()
sampler = Sampler(model, tokenizer)

prompts = [
    "How do I implement quicksort in Python?",
    "Write a short poem about the ocean.",
    "What is the best shipment route for our cargo?",
    "What is the capital of France?",
]

for prompt in prompts:
    profile = router.route(prompt)
    print(f"\n{'─'*60}")
    print(f"Prompt   : {prompt}")
    router.describe(prompt)

    response = sampler.sample(
        prompt,
        max_new_tokens=profile.max_new_tokens,
        sampling=profile.sampling,
    )
    print(f"Response : {response!r}")
