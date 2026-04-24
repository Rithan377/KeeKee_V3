"""
chat_keekee.py — Interactive chat with KeeKee (SODA-trained)
=============================================================
Loads best.pt from TRAIN/checkpoints/ and starts a multi-turn
conversation loop in the terminal with live streaming output.

Usage:
    python chat_keekee.py
    python chat_keekee.py --checkpoint TRAIN/checkpoints/best.pt
    python chat_keekee.py --max_new_tokens 256 --temperature 0.8
    python chat_keekee.py --cpu     # force CPU if GPU is busy
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import torch
import torch.nn.functional as F
from dataclasses import replace as dataclass_replace
from pathlib import Path

from config import MODEL_130M
from model import KeeKee
from tokenizer import BPETokenizer


# ─────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(
    model: KeeKee,
    input_ids: torch.Tensor,         # [1, L]
    tokenizer: BPETokenizer,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    stream: bool = True,
) -> str:
    """
    Autoregressive generation with top-p + top-k sampling.
    Streams tokens to stdout in real time.
    Stops at <end_of_turn> or <eos>.
    """
    model.eval()
    device   = input_ids.device
    stop_ids = tokenizer.end_tokens()   # {eos_id, end_turn_id}

    generated = []
    current_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        out    = model(current_ids)
        logits = out.logits[0, -1, :]

        # Repetition penalty
        if repetition_penalty != 1.0 and generated:
            for tok_id in set(generated):
                logits[tok_id] /= repetition_penalty

        # Temperature
        logits = logits / max(temperature, 1e-8)

        # Top-k
        if top_k > 0:
            top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_k_vals[-1]] = float("-inf")

        # Top-p (nucleus)
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_logits[cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p] = float("-inf")
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(0, sorted_idx, sorted_logits)

        probs    = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        tok_id   = next_tok.item()

        if tok_id in stop_ids:
            break

        generated.append(tok_id)

        # Stream: decode and print just the new token
        if stream:
            word = tokenizer.decode([tok_id])
            print(word, end="", flush=True)

        # Keep context window to last 1024 tokens
        current_ids = torch.cat([
            current_ids[:, -1023:],
            next_tok.unsqueeze(0)
        ], dim=1)

    if stream:
        print()   # newline after streamed reply

    return tokenizer.decode(generated)


# ─────────────────────────────────────────────────────────────────
# Build prompt from conversation history
# ─────────────────────────────────────────────────────────────────

# ── Identity primer injected silently at the start of every conversation ──
# This grounds the model as KeeKee without relying purely on fine-tuned weights.
# Multiple primer turns reinforce the name and AI identity strongly.
IDENTITY_PRIMER: list[tuple[str, str]] = [
    ("what is your name",
     "My name is KeeKee. I'm your AI assistant!"),
    ("are you human",
     "No, I'm KeeKee — an AI assistant. Not human, but here to help!"),
    ("who are you",
     "I'm KeeKee, an AI assistant. I'm here to chat and help with anything you need."),
]

# ── The greeting shown to the user on startup ──
STARTUP_GREETING = "Hi! I'm KeeKee, your AI assistant. How can I help you today?"


def build_prompt(
    tokenizer: BPETokenizer,
    history: list[tuple[str, str]],   # [(user_msg, model_reply), ...]
    new_user_msg: str,
    max_history_turns: int = 6,
) -> torch.Tensor:
    """
    Builds the token sequence for the next model turn:
      <bos> [<user>\nMSG<end> <model>\nREPLY<end>]* <user>\nMSG<end> <model>
    The model then generates from the trailing <model> token onwards.

    A silent identity primer is always prepended so KeeKee knows its name
    and AI nature regardless of the base model weights.
    """
    ids = [tokenizer.bos_id]

    # Always prepend identity primer turns (invisible to the user)
    for p_user, p_asst in IDENTITY_PRIMER:
        ids += tokenizer.encode_chat_turn("user",      p_user)
        ids += tokenizer.encode_chat_turn("assistant", p_asst)

    # Inject recent history (capped so primer + history fit in context)
    primer_turns = len(IDENTITY_PRIMER)
    for user_msg, model_reply in history[-(max_history_turns - primer_turns):]:
        ids += tokenizer.encode_chat_turn("user",      user_msg)
        ids += tokenizer.encode_chat_turn("assistant", model_reply)

    # New user turn
    ids += tokenizer.encode_chat_turn("user", new_user_msg)

    # Open the model turn — DO NOT add \n here, encode_chat_turn already adds it
    ids += [tokenizer.model_id]

    return torch.tensor([ids], dtype=torch.long)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Chat with KeeKee")
    p.add_argument("--checkpoint",         default="TRAIN/checkpoints/best.pt")
    p.add_argument("--max_new_tokens",     type=int,   default=200)
    p.add_argument("--temperature",        type=float, default=0.75)
    p.add_argument("--top_p",             type=float, default=0.92)
    p.add_argument("--top_k",             type=int,   default=50)
    p.add_argument("--repetition_penalty", type=float, default=1.3)
    p.add_argument("--history_turns",      type=int,   default=6)
    p.add_argument("--cpu",               action="store_true", help="Force CPU inference")
    p.add_argument("--no_stream",         action="store_true", help="Disable token streaming")
    args = p.parse_args()

    # ── Device ──────────────────────────────────────────────────
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    W = 58
    print(f"\n{'─'*W}")
    print(f"  🤖  KeeKee v3  —  AI Chat Assistant")
    print(f"{'─'*W}")
    print(f"  Device     : {device}")

    # ── Tokenizer ───────────────────────────────────────────────
    tokenizer = BPETokenizer()
    print(f"  Tokenizer  : BPE  |  vocab = {tokenizer.vocab_size:,}")

    # ── Model config ────────────────────────────────────────────
    cfg = MODEL_130M
    if cfg.vocab_size != tokenizer.vocab_size:
        cfg = dataclass_replace(cfg, vocab_size=tokenizer.vocab_size)
    cfg = dataclass_replace(cfg, dropout=0.0)

    # ── Load checkpoint ─────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"\n  ❌  Checkpoint not found: {ckpt_path}")
        print(f"      Train first: python TRAIN/train.py")
        sys.exit(1)

    ckpt  = torch.load(ckpt_path, map_location=device)
    model = KeeKee(cfg).to(device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    best_val  = ckpt.get("best_val_loss", "?")
    step      = ckpt.get("step", "?")
    print(f"  Checkpoint : {ckpt_path.name}  |  step {step:,}")
    print(f"  Parameters : {n_params:.1f}M  |  val_loss {best_val:.4f}")

    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU        : {torch.cuda.get_device_name(0)} ({vram:.1f} GB)")

    print(f"{'─'*W}")

    # ── Chat loop ───────────────────────────────────────────────
    print("  Commands : 'quit' | 'clear' to reset | 'history' to show context")
    print(f"  Streaming : {'OFF' if args.no_stream else 'ON'}")
    print(f"{'─'*W}\n")

    # ── KeeKee greets the user on startup ───────────────────────
    # Printed visibly so the user immediately knows who they're talking to.
    # Also seeded into history so the model "remembers" it already introduced itself.
    print(f"\033[1;32mKeeKee\033[0m: {STARTUP_GREETING}\n")

    history: list[tuple[str, str]] = [
        ("hi", STARTUP_GREETING)
    ]

    while True:
        try:
            user_input = input("\033[1;36mYou\033[0m: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye! 👋")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("quit", "exit", "q"):
            print("  Goodbye! 👋")
            break
        if cmd == "clear":
            # Reset history but re-seed the greeting so identity is preserved
            history = [("hi", STARTUP_GREETING)]
            print("  ✅  Conversation history cleared.\n")
            continue
        if cmd == "history":
            # Skip the seeded greeting turn (index 0) when displaying to user
            visible_history = history[1:]
            if not visible_history:
                print("  (no history yet)\n")
            else:
                print(f"  ── {len(visible_history)} turn(s) in context ──")
                for i, (u, m) in enumerate(visible_history, 1):
                    print(f"  [{i}] You     : {u[:60]}")
                    print(f"       KeeKee  : {m[:60]}")
                print()
            continue

        # Build full prompt tensor
        input_ids = build_prompt(
            tokenizer, history, user_input,
            max_history_turns=args.history_turns,
        ).to(device)

        # Generate — streams token by token
        print("\033[1;32mKeeKee\033[0m: ", end="", flush=True)
        response = generate(
            model, input_ids, tokenizer,
            max_new_tokens     = args.max_new_tokens,
            temperature        = args.temperature,
            top_p              = args.top_p,
            top_k              = args.top_k,
            repetition_penalty = args.repetition_penalty,
            stream             = not args.no_stream,
        )
        if args.no_stream:
            print(response.strip())
        print()

        # Save to history
        history.append((user_input, response.strip()))


if __name__ == "__main__":
    main()