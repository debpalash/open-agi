"""
Direction 1+2: Process Value Function + Continual Self-Improvement

Novel training loop:
  1. Load pretrained novel model
  2. Generate solutions for problems with known answers
  3. Score each solution (correct/incorrect) 
  4. Train a value head to predict correctness from hidden states
  5. Use value-weighted loss to retrain the LM
  6. Iterate (self-improvement loop)

This combines Ilya's insights:
  - Value function for intermediate reasoning evaluation
  - Self-improvement through filtered self-generation
"""

import gc, math, os, re, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from prepare import MAX_SEQ_LEN, Tokenizer
from prepare_structured import TOPIC_MAP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = Tokenizer()

# ── Model (same as benchmark.py but with value head) ─────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
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

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim//n_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        k = self.k(x).view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        v = self.v(x).view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        q, k = _apply_rope(q,cos,sin), _apply_rope(k,cos,sin)
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.o(y.transpose(1,2).contiguous().view(B,T,-1))

class DenseFFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.c_fc = nn.Linear(dim, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())

class TopicConditionedFFN(nn.Module):
    def __init__(self, dim, hidden, topic_embed_dim=64):
        super().__init__()
        self.c_fc = nn.Linear(dim, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, dim, bias=False)
        self.topic_bias = nn.Linear(topic_embed_dim, hidden, bias=False)
    def forward(self, x, topic_emb=None):
        h = self.c_fc(x)
        if topic_emb is not None:
            h = h + self.topic_bias(topic_emb).unsqueeze(1)
        return self.c_proj(F.relu(h).square())

class Block(nn.Module):
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


class PVFModel(nn.Module):
    """
    NOVEL: Process Value Function Model
    
    Same as NovelRDT but with an additional value head that predicts
    P(correct_final_answer | hidden_state_at_position).
    
    This is an INTERNAL value function — the model evaluates its own
    reasoning quality as it reasons, not as a separate critic model.
    """
    def __init__(self, vocab_size, dim=512, n_heads=8, prelude=2, coda=2, 
                 n_loops=1, ffn_mult=4, n_topics=8, topic_embed_dim=64):
        super().__init__()
        self.n_loops = n_loops
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.prelude = nn.ModuleList([Block(dim,n_heads,ffn_mult,False) for _ in range(prelude)])
        self.core = Block(dim, n_heads, ffn_mult, use_topic=True)
        self.coda = nn.ModuleList([Block(dim,n_heads,ffn_mult,True) for _ in range(coda)])
        self.norm_out = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Topic classifier + embedding (from novel model)
        self.topic_classifier = nn.Sequential(
            nn.Linear(dim, topic_embed_dim), nn.ReLU(), nn.Linear(topic_embed_dim, n_topics))
        self.topic_embedding = nn.Embedding(n_topics, topic_embed_dim)
        
        # NOVEL: Process Value Head
        # Predicts P(correct_answer) from hidden state at any position
        self.value_head = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
        )
        
        cos, sin = _precompute_rope(MAX_SEQ_LEN, dim//n_heads, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    def forward(self, idx, targets=None):
        B,T = idx.shape
        cos,sin = self.cos[:,:,:T], self.sin[:,:,:T]
        x = self.tok_emb(idx)
        for b in self.prelude: x = b(x, cos, sin)
        
        pooled = x[:,:min(T,64),:].mean(dim=1)
        tl = self.topic_classifier(pooled)
        tid = tl.argmax(dim=-1)
        topic_emb = self.topic_embedding(tid)
        
        for _ in range(self.n_loops): x = self.core(x, cos, sin, topic_emb)
        for b in self.coda: x = b(x, cos, sin, topic_emb)
        
        h = self.norm_out(x)
        logits = self.lm_head(h)
        
        # Value prediction: mean pool over all positions
        value = self.value_head(h.mean(dim=1)).squeeze(-1)  # (B,)
        
        if targets is not None:
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return lm_loss, logits, value
        return logits, value
    
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Load pretrained weights ──────────────────────────────────────────────

def load_pretrained():
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", "novel_rdt.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    
    model = PVFModel(tok.get_vocab_size()).to(device)
    # Load matching keys, skip value_head (new)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded pretrained: {len(sd)} keys, {len(missing)} new (value_head), {len(unexpected)} skipped")
    return model


# ── Generation ───────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, prompt_ids, max_new=300, temperature=0.5, top_k=30):
    model.eval()
    ids = torch.tensor([prompt_ids], device=device)
    for _ in range(max_new):
        ctx = ids[:, -MAX_SEQ_LEN:]
        out = model(ctx)
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits[:, -1, :] / temperature
        if top_k:
            v, _ = logits.topk(top_k)
            logits[logits < v[:, -1:]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
        if tok.decode(ids[0].tolist()[-8:]).count("<|end|>") > 0:
            break
    return ids[0].tolist()


# ── Answer extraction + verification ─────────────────────────────────────

def extract_answer(text):
    """Extract answer from model output."""
    # Look for <answer> X pattern
    m = re.search(r'<answer>\s*(.+?)(?:\s*<|$)', text)
    if m:
        return m.group(1).strip()
    # Look for "the answer is X" pattern
    m = re.search(r'(?:answer|result)\s+(?:is|=)\s+[\\$]*(.+?)[\\$]*(?:\.|,|$)', text, re.I)
    if m:
        return m.group(1).strip()
    return None

def normalize_answer(ans):
    """Normalize answer for comparison."""
    if ans is None:
        return None
    ans = ans.strip().lower()
    ans = re.sub(r'[\\${}]', '', ans)
    ans = re.sub(r'\s+', ' ', ans)
    # Try to evaluate simple expressions
    ans = ans.replace('\\frac', '').replace('\\cdot', '*')
    return ans

def answers_match(predicted, expected):
    """Check if predicted answer matches expected."""
    if predicted is None or expected is None:
        return False
    p, e = normalize_answer(predicted), normalize_answer(expected)
    if p == e:
        return True
    # Try numeric comparison
    try:
        pv = eval(p.replace('^', '**'))
        ev = eval(e.replace('^', '**'))
        return abs(pv - ev) < 1e-6
    except:
        pass
    # Check if one contains the other
    return p in e or e in p


# ── Self-Improvement Loop ────────────────────────────────────────────────

def run_self_improvement(n_iterations=3, problems_per_iter=200, samples_per_problem=3):
    """
    The core self-improvement loop:
    1. Generate solutions
    2. Verify against ground truth
    3. Collect (correct_solution, value=1) and (wrong_solution, value=0)
    4. Train value head + value-weighted LM loss
    5. Repeat
    """
    model = load_pretrained()
    print(f"Model: {model.num_params()/1e6:.1f}M parameters")
    
    from datasets import load_dataset
    ds = load_dataset("ShadenA/MathNet", split="train", streaming=True)
    
    # Collect problems with verifiable answers
    problems = []
    for row in ds:
        ans = row.get("final_answer")
        prob = row.get("problem_markdown", "") or ""
        topics = row.get("topics_flat", []) or []
        if not ans or not ans.strip() or len(prob) > 500:
            continue
        top = topics[0].split(" > ")[0] if topics else "Algebra"
        problems.append({"problem": prob, "answer": ans.strip(), "topic": top})
        if len(problems) >= problems_per_iter * n_iterations:
            break
    
    print(f"Collected {len(problems)} verifiable problems")
    
    for iteration in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{n_iterations}")
        print(f"{'='*60}")
        
        # ── Phase 1: Generate solutions ──
        print("\n[Phase 1] Generating solutions...")
        correct_solutions = []
        wrong_solutions = []
        start = iteration * problems_per_iter
        batch = problems[start:start + problems_per_iter]
        
        model.eval()
        t0 = time.time()
        for i, p in enumerate(batch):
            prompt = f"<topic> {p['topic']}\n<|problem|>\n{p['problem']}\n<|solution|>\n"
            prompt_ids = tok.encode(prompt)
            
            for s in range(samples_per_problem):
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_ids = generate(model, prompt_ids, max_new=250, 
                                      temperature=0.4 + 0.2*s)  # vary temperature
                
                out_text = tok.decode(out_ids[len(prompt_ids):])
                pred_ans = extract_answer(out_text)
                is_correct = answers_match(pred_ans, p["answer"])
                
                full_ids = out_ids  # prompt + solution
                entry = {
                    "ids": full_ids,
                    "correct": is_correct,
                    "pred": pred_ans,
                    "expected": p["answer"],
                }
                
                if is_correct:
                    correct_solutions.append(entry)
                else:
                    wrong_solutions.append(entry)
            
            if (i+1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{len(batch)} problems | "
                      f"correct: {len(correct_solutions)} | wrong: {len(wrong_solutions)} | "
                      f"{elapsed:.0f}s")
        
        total = len(correct_solutions) + len(wrong_solutions)
        acc = 100 * len(correct_solutions) / max(total, 1)
        print(f"\n  Results: {len(correct_solutions)} correct / {total} total ({acc:.1f}%)")
        
        if len(correct_solutions) == 0:
            print("  No correct solutions found. Continuing with value head training only...")
        
        # ── Phase 2: Train value head ──
        print("\n[Phase 2] Training value head + value-weighted LM...")
        model.train()
        
        # Create training data: mix of correct (value=1) and wrong (value=0)
        all_solutions = [(s, 1.0) for s in correct_solutions] + [(s, 0.0) for s in wrong_solutions]
        np.random.shuffle(all_solutions)
        
        # Only train for a short time
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        n_train_steps = min(len(all_solutions) * 2, 200)  # cap at 200 steps
        
        total_lm_loss = 0
        total_value_loss = 0
        n_batched = 0
        
        for step in range(n_train_steps):
            idx = step % len(all_solutions)
            sol, value_target = all_solutions[idx]
            ids = sol["ids"]
            
            # Truncate to MAX_SEQ_LEN
            if len(ids) > MAX_SEQ_LEN + 1:
                ids = ids[:MAX_SEQ_LEN + 1]
            if len(ids) < 10:
                continue
            
            x = torch.tensor([ids[:-1]], device=device, dtype=torch.long)
            y = torch.tensor([ids[1:]], device=device, dtype=torch.long)
            vt = torch.tensor([value_target], device=device, dtype=torch.float32)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                lm_loss, logits, value_pred = model(x, y)
                
                # Value loss: binary cross-entropy
                value_loss = F.binary_cross_entropy_with_logits(value_pred, vt)
                
                # VALUE-WEIGHTED LM loss: correct solutions get higher weight
                lm_weight = 2.0 if value_target > 0.5 else 0.5
                
                loss = lm_weight * lm_loss + 0.5 * value_loss
            
            loss.backward()
            
            if (step + 1) % 4 == 0:  # gradient accumulation
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            total_lm_loss += lm_loss.item()
            total_value_loss += value_loss.item()
            n_batched += 1
            
            if (step + 1) % 50 == 0:
                print(f"  step {step+1}/{n_train_steps} | "
                      f"lm: {total_lm_loss/n_batched:.4f} | "
                      f"value: {total_value_loss/n_batched:.4f}")
        
        if n_batched > 0:
            print(f"  Final: lm={total_lm_loss/n_batched:.4f} value={total_value_loss/n_batched:.4f}")
        
        # ── Phase 3: Evaluate ──
        print("\n[Phase 3] Evaluating after training...")
        model.eval()
        eval_correct = 0
        eval_total = 0
        eval_problems = problems[:50]  # test on first 50
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            for p in eval_problems:
                prompt = f"<topic> {p['topic']}\n<|problem|>\n{p['problem']}\n<|solution|>\n"
                prompt_ids = tok.encode(prompt)
                out_ids = generate(model, prompt_ids, max_new=250, temperature=0.3)
                out_text = tok.decode(out_ids[len(prompt_ids):])
                pred_ans = extract_answer(out_text)
                if answers_match(pred_ans, p["answer"]):
                    eval_correct += 1
                eval_total += 1
        
        eval_acc = 100 * eval_correct / max(eval_total, 1)
        print(f"  Accuracy: {eval_correct}/{eval_total} = {eval_acc:.1f}%")
        
        # ── Phase 4: Value head accuracy ──
        print("\n[Phase 4] Value head calibration...")
        value_correct = 0
        value_total = 0
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            for sol, target in all_solutions[:100]:
                ids = sol["ids"][:MAX_SEQ_LEN]
                if len(ids) < 10:
                    continue
                x = torch.tensor([ids], device=device, dtype=torch.long)
                _, value_pred = model(x)
                pred_label = (torch.sigmoid(value_pred) > 0.5).float().item()
                if pred_label == target:
                    value_correct += 1
                value_total += 1
        
        val_acc = 100 * value_correct / max(value_total, 1)
        print(f"  Value head accuracy: {value_correct}/{value_total} = {val_acc:.1f}%")
    
    # Save final model
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "pvf_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"type": "PVFModel", "novel_features": [
            "process_value_function", "topic_conditioned_ffn", 
            "value_weighted_loss", "self_improvement_loop"
        ]},
    }, ckpt_path)
    print(f"\nSaved: {ckpt_path} ({os.path.getsize(ckpt_path)/1e6:.1f} MB)")
    
    return model


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    print("=" * 60)
    print("DIRECTION 1+2: Process Value Function + Self-Improvement")
    print("=" * 60)
    run_self_improvement(n_iterations=3, problems_per_iter=100, samples_per_problem=2)
