"""
Benchmark: Baseline vs Novel model — inference + math evaluation.
Usage: uv run benchmark.py
"""
import os, sys, time, math, torch, numpy as np
import torch.nn.functional as F

# Add parent for imports
sys.path.insert(0, os.path.dirname(__file__))
from prepare import MAX_SEQ_LEN, Tokenizer

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = Tokenizer()

# ── Model Definitions (must match training) ──────────────────────────────

class RMSNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, 1e-6)

def _apply_rope(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)

def _precompute_rope(seq_len, head_dim, device, base=10000.0):
    inv = 1.0/(base**(torch.arange(0, head_dim, 2, device=device).float()/head_dim))
    t = torch.arange(seq_len, device=device).float()
    f = torch.outer(t, inv)
    return f.cos()[None,None,:,:], f.sin()[None,None,:,:]

class Attention(torch.nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim//n_heads
        self.q = torch.nn.Linear(dim, dim, bias=False)
        self.k = torch.nn.Linear(dim, dim, bias=False)
        self.v = torch.nn.Linear(dim, dim, bias=False)
        self.o = torch.nn.Linear(dim, dim, bias=False)
    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        k = self.k(x).view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        v = self.v(x).view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        q, k = _apply_rope(q,cos,sin), _apply_rope(k,cos,sin)
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.o(y.transpose(1,2).contiguous().view(B,T,-1))

class DenseFFN(torch.nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.c_fc = torch.nn.Linear(dim, hidden, bias=False)
        self.c_proj = torch.nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())

# ── Baseline RDT ─────────────────────────────────────────────────────────

class BaselineBlock(torch.nn.Module):
    def __init__(self, dim, n_heads, ffn_mult):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        hidden = ((int(dim*ffn_mult)+63)//64)*64
        self.ffn = DenseFFN(dim, hidden)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x

class BaselineCore(torch.nn.Module):
    def __init__(self, dim, n_heads, ffn_mult, max_loops):
        super().__init__()
        self.block = BaselineBlock(dim, n_heads, ffn_mult)
        self.loop_embed = torch.nn.Embedding(max_loops, dim) if max_loops > 1 else None
    def forward(self, x, e, cos, sin, n_loops):
        for i in range(n_loops):
            x_in = x
            if self.loop_embed is not None:
                x_in = x_in + self.loop_embed(torch.tensor(i, device=x.device))
            x = self.block(x_in, cos, sin)
        return x

class BaselineRDT(torch.nn.Module):
    def __init__(self, vocab_size, dim=512, n_heads=8, prelude=2, coda=2, n_loops=1, ffn_mult=4):
        super().__init__()
        self.n_loops = n_loops
        self.tok_emb = torch.nn.Embedding(vocab_size, dim)
        self.prelude = torch.nn.ModuleList([BaselineBlock(dim,n_heads,ffn_mult) for _ in range(prelude)])
        self.core = BaselineCore(dim, n_heads, ffn_mult, max(n_loops,32))
        self.coda = torch.nn.ModuleList([BaselineBlock(dim,n_heads,ffn_mult) for _ in range(coda)])
        self.norm_out = RMSNorm(dim)
        self.lm_head = torch.nn.Linear(dim, vocab_size, bias=False)
        cos, sin = _precompute_rope(MAX_SEQ_LEN, dim//n_heads, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    def forward(self, idx, targets=None):
        B,T = idx.shape; cos,sin = self.cos[:,:,:T], self.sin[:,:,:T]
        x = self.tok_emb(idx); e = x
        for b in self.prelude: x = b(x, cos, sin)
        x = self.core(x, e, cos, sin, self.n_loops)
        for b in self.coda: x = b(x, cos, sin)
        logits = self.lm_head(self.norm_out(x))
        if targets is not None:
            return F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1))
        return logits

# ── Novel RDT ────────────────────────────────────────────────────────────

class TopicConditionedFFN(torch.nn.Module):
    def __init__(self, dim, hidden, n_topics=8, topic_embed_dim=64):
        super().__init__()
        self.c_fc = torch.nn.Linear(dim, hidden, bias=False)
        self.c_proj = torch.nn.Linear(hidden, dim, bias=False)
        self.topic_bias = torch.nn.Linear(topic_embed_dim, hidden, bias=False)
    def forward(self, x, topic_emb=None):
        h = self.c_fc(x)
        if topic_emb is not None:
            h = h + self.topic_bias(topic_emb).unsqueeze(1)
        return self.c_proj(F.relu(h).square())

class NovelBlock(torch.nn.Module):
    def __init__(self, dim, n_heads, ffn_mult, use_topic=False):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        hidden = ((int(dim*ffn_mult)+63)//64)*64
        self.use_topic = use_topic
        self.ffn = TopicConditionedFFN(dim, hidden) if use_topic else DenseFFN(dim, hidden)
    def forward(self, x, cos, sin, topic_emb=None):
        x = x + self.attn(self.norm1(x), cos, sin)
        if self.use_topic and topic_emb is not None:
            x = x + self.ffn(self.norm2(x), topic_emb)
        else:
            x = x + self.ffn(self.norm2(x))
        return x

class NovelRDT(torch.nn.Module):
    def __init__(self, vocab_size, dim=512, n_heads=8, prelude=2, coda=2, n_loops=1, ffn_mult=4):
        super().__init__()
        self.n_loops = n_loops
        self.tok_emb = torch.nn.Embedding(vocab_size, dim)
        self.prelude = torch.nn.ModuleList([NovelBlock(dim,n_heads,ffn_mult,False) for _ in range(prelude)])
        self.core = NovelBlock(dim, n_heads, ffn_mult, use_topic=True)
        self.coda = torch.nn.ModuleList([NovelBlock(dim,n_heads,ffn_mult,True) for _ in range(coda)])
        self.norm_out = RMSNorm(dim)
        self.lm_head = torch.nn.Linear(dim, vocab_size, bias=False)
        self.topic_classifier = torch.nn.Sequential(
            torch.nn.Linear(dim, 64), torch.nn.ReLU(), torch.nn.Linear(64, 8))
        self.topic_embedding = torch.nn.Embedding(8, 64)
        cos, sin = _precompute_rope(MAX_SEQ_LEN, dim//n_heads, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    def forward(self, idx, targets=None, topic_labels=None, verify_mask=None):
        B,T = idx.shape; cos,sin = self.cos[:,:,:T], self.sin[:,:,:T]
        x = self.tok_emb(idx)
        for b in self.prelude: x = b(x, cos, sin)
        pooled = x[:,:min(T,64),:].mean(dim=1)
        tl = self.topic_classifier(pooled)
        tid = tl.argmax(dim=-1)
        topic_emb = self.topic_embedding(tid)
        for _ in range(self.n_loops): x = self.core(x, cos, sin, topic_emb)
        for b in self.coda: x = b(x, cos, sin, topic_emb)
        logits = self.lm_head(self.norm_out(x))
        if targets is not None:
            lm_loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1))
            topic_loss = torch.tensor(0.0, device=idx.device)
            return lm_loss, topic_loss, logits
        return logits

# ── Generation ───────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, prompt_ids, max_new=256, temperature=0.7, top_k=50):
    model.eval()
    ids = torch.tensor([prompt_ids], device=device)
    for _ in range(max_new):
        ctx = ids[:, -MAX_SEQ_LEN:]
        out = model(ctx)
        if isinstance(out, tuple): out = out[-1]
        logits = out[:, -1, :] / temperature
        if top_k:
            v, _ = logits.topk(top_k)
            logits[logits < v[:, -1:]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
        # Stop on <|end|>
        decoded = tok.decode([nxt.item()])
        if "<|end|>" in tok.decode(ids[0].tolist()[-10:]):
            break
    return ids[0].tolist()

# ── Perplexity on held-out problems ─────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, data_tokens, n_windows=50):
    model.eval()
    total_loss, total_toks = 0.0, 0
    for _ in range(n_windows):
        i = np.random.randint(0, len(data_tokens) - MAX_SEQ_LEN - 1)
        x = torch.tensor(data_tokens[i:i+MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(data_tokens[i+1:i+MAX_SEQ_LEN+1], dtype=torch.long, device=device).unsqueeze(0)
        out = model(x)
        if isinstance(out, tuple): out = out[-1]
        loss = F.cross_entropy(out.view(-1, out.size(-1)), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_toks += y.numel()
    return math.exp(total_loss / total_toks)

# ── Main ─────────────────────────────────────────────────────────────────

def _strip_compile_prefix(sd):
    """Remove _orig_mod. prefix added by torch.compile."""
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

def load_baseline():
    path = os.path.join(CKPT_DIR, "rdt_best.pt")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = _strip_compile_prefix(ckpt.get("model_state_dict", ckpt))
    m = BaselineRDT(tok.get_vocab_size()).to(device)
    m.load_state_dict(sd, strict=False)
    return m

def load_novel():
    path = os.path.join(CKPT_DIR, "novel_rdt.pt")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = _strip_compile_prefix(ckpt.get("model_state_dict", ckpt))
    m = NovelRDT(tok.get_vocab_size()).to(device)
    m.load_state_dict(sd, strict=False)
    return m

# Test problems
PROBLEMS = [
    "Find all positive integers $n$ such that $n^2 + 1$ divides $n! + 1$.",
    "Let $ABC$ be a triangle with $AB = 13$, $BC = 14$, $CA = 15$. Find the area of triangle $ABC$.",
    "Prove that for all positive reals $a, b, c$: $\\frac{a}{b+c} + \\frac{b}{a+c} + \\frac{c}{a+b} \\geq \\frac{3}{2}$.",
    "How many ways can you tile a $2 \\times 10$ board with $1 \\times 2$ dominoes?",
    "Find the remainder when $2^{100}$ is divided by $7$.",
]

TOPIC_NAMES = ["Geometry", "Algebra", "Discrete Mathematics", "Number Theory",
               "Statistics", "Calculus", "Precalculus", "Math Word Problems"]

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("BENCHMARK: Baseline RDT vs Novel TopicConditioned RDT")
    print("=" * 70)

    # Load models
    print("\nLoading Baseline (67.2M)...")
    baseline = load_baseline()
    print("Loading Novel (67.6M)...")
    novel = load_novel()

    # Load val data for perplexity
    from prepare_structured import STRUCTURED_DIR
    val_path = os.path.join(STRUCTURED_DIR, "val.bin")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
    std_val = os.path.expanduser("~/.cache/autoresearch-mythos/val.bin")
    std_data = np.memmap(std_val, dtype=np.uint16, mode="r") if os.path.exists(std_val) else val_data

    # 1. Perplexity
    print("\n── Perplexity (lower=better) ──")
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        ppl_base_std = compute_perplexity(baseline, std_data, n_windows=80)
        ppl_base_struct = compute_perplexity(baseline, val_data, n_windows=80)
        ppl_novel_std = compute_perplexity(novel, std_data, n_windows=80)
        ppl_novel_struct = compute_perplexity(novel, val_data, n_windows=80)

    print(f"{'':30s} {'Standard':>12s} {'Structured':>12s}")
    print(f"{'Baseline RDT':30s} {ppl_base_std:12.2f} {ppl_base_struct:12.2f}")
    print(f"{'Novel TopicCond RDT':30s} {ppl_novel_std:12.2f} {ppl_novel_struct:12.2f}")

    # 2. Topic classification accuracy (Novel only)
    print("\n── Topic Classification (Novel model) ──")
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        novel.eval()
        from datasets import load_dataset
        ds = load_dataset("ShadenA/MathNet", split="train", streaming=True)
        correct, total = 0, 0
        from prepare_structured import TOPIC_MAP
        for row in ds:
            if total >= 200: break
            topics = row.get("topics_flat", []) or []
            if not topics: continue
            top = topics[0].split(" > ")[0]
            if top not in TOPIC_MAP: continue
            gt = TOPIC_MAP[top]
            prob_text = row.get("problem_markdown", "") or ""
            ids = tok.encode(prob_text)[:MAX_SEQ_LEN]
            if len(ids) < 10: continue
            x = torch.tensor([ids], device=device)
            # Run through prelude + classifier
            cos, sin = novel.cos[:,:,:len(ids)], novel.sin[:,:,:len(ids)]
            h = novel.tok_emb(x)
            for b in novel.prelude: h = b(h, cos, sin)
            pooled = h[:,:min(len(ids),64),:].mean(dim=1)
            pred = novel.topic_classifier(pooled).argmax(dim=-1).item()
            if pred == gt: correct += 1
            total += 1
        acc = 100 * correct / max(total, 1)
        print(f"Accuracy: {correct}/{total} = {acc:.1f}%")
        print(f"(Random baseline: {100/len(TOPIC_MAP):.1f}%)")

    # 3. Generation comparison
    print("\n── Generation Comparison ──")
    for i, prob in enumerate(PROBLEMS):
        print(f"\n{'─'*60}")
        print(f"Problem {i+1}: {prob[:80]}...")
        prompt = f"<|problem|>\n{prob}\n<|solution|>\n"
        prompt_ids = tok.encode(prompt)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            t0 = time.time()
            base_out = generate(baseline, prompt_ids, max_new=200)
            base_time = time.time() - t0
            base_text = tok.decode(base_out[len(prompt_ids):])

            t0 = time.time()
            novel_out = generate(novel, prompt_ids, max_new=200)
            novel_time = time.time() - t0
            novel_text = tok.decode(novel_out[len(prompt_ids):])

        print(f"\n  Baseline ({base_time:.2f}s):")
        print(f"    {base_text[:300]}")
        print(f"\n  Novel ({novel_time:.2f}s):")
        print(f"    {novel_text[:300]}")

    # 4. Inference speed
    print(f"\n{'─'*60}")
    print("── Inference Speed ──")
    prompt_ids = tok.encode("<|problem|>\nFind x if $2x + 3 = 11$.\n<|solution|>\n")
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Warmup
        generate(baseline, prompt_ids, max_new=50)
        generate(novel, prompt_ids, max_new=50)
        # Timed
        t0 = time.time()
        for _ in range(5): generate(baseline, prompt_ids, max_new=100)
        base_speed = (time.time()-t0)/5
        t0 = time.time()
        for _ in range(5): generate(novel, prompt_ids, max_new=100)
        novel_speed = (time.time()-t0)/5
    print(f"Baseline: {base_speed:.3f}s / 100 tokens")
    print(f"Novel:    {novel_speed:.3f}s / 100 tokens")
    print(f"Overhead: {(novel_speed/base_speed - 1)*100:+.1f}%")

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")
