"""
Structured Math Data Preparation for Novel Research.

Transforms raw MathNet into structured training data with:
  1. Step-decomposed solutions with <step> / <verify> / <correct> tokens
  2. Topic classification labels (top-level + fine-grained)
  3. Final answer verification signal

Usage:
    python prepare_structured.py           # prepare structured data
    python prepare_structured.py --stats   # show dataset statistics
"""

import os
import re
import math
import numpy as np
from typing import Optional
from prepare import DATA_DIR, MAX_SEQ_LEN, VAL_RATIO, Tokenizer

STRUCTURED_DIR = os.path.join(DATA_DIR, "structured")

# Special tokens — we'll add these to the tokenizer's vocab
SPECIAL_TOKENS = {
    "<step>":     50257,   # step boundary marker
    "<verify>":   50258,   # verification checkpoint
    "<correct>":  50259,   # step verified correct
    "<revise>":   50260,   # step needs revision (used in negative examples)
    "<topic>":    50261,   # topic label marker
    "<answer>":   50262,   # final answer marker
    "<|problem|>": None,   # already in vocab via tiktoken (use existing)
    "<|solution|>": None,
    "<|end|>": None,
}

# Step boundary patterns — regex patterns that indicate logical step transitions
STEP_PATTERNS = [
    r'\n\n+',                          # double newline (most common)
    r'(?<=\.)\s*(?=Therefore)',        # before "Therefore"
    r'(?<=\.)\s*(?=Thus)',             # before "Thus"
    r'(?<=\.)\s*(?=Hence)',            # before "Hence"
    r'(?<=\.)\s*(?=So\s)',             # before "So"
    r'(?<=\.)\s*(?=It follows)',       # before "It follows"
    r'(?<=\.)\s*(?=We conclude)',      # before "We conclude"
    r'(?<=\.)\s*(?=This gives)',       # before "This gives"
    r'(?<=\.)\s*(?=Consequently)',     # before "Consequently"
    r'(?<=\$\$)\s*',                   # after display math
    r'(?<=\\square)\s*',              # after QED
]

# Topic taxonomy — map fine-grained topics to top-level categories
TOPIC_MAP = {
    "Geometry": 0,
    "Algebra": 1,
    "Discrete Mathematics": 2,
    "Number Theory": 3,
    "Statistics": 4,
    "Calculus": 5,
    "Precalculus": 6,
    "Math Word Problems": 7,
}


def parse_steps(solution: str) -> list[str]:
    """
    Parse a math solution into logical reasoning steps.
    
    Returns a list of non-empty step strings.
    """
    if not solution or len(solution.strip()) < 10:
        return [solution.strip()] if solution else []
    
    # First split on double newlines (preserves paragraph structure)
    parts = re.split(r'\n\n+', solution)
    
    steps = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Further split long paragraphs on logical transition words
        if len(part) > 500:
            sub_parts = re.split(
                r'(?<=\.)\s+(?=(?:Therefore|Thus|Hence|So|It follows|We conclude|This gives|Consequently)\b)',
                part
            )
            steps.extend(s.strip() for s in sub_parts if s.strip())
        else:
            steps.append(part)
    
    # If we only got 1 step and it's long, try splitting on sentences ending with $$
    if len(steps) == 1 and len(steps[0]) > 300:
        sub = re.split(r'(?<=\$\$)\s+', steps[0])
        if len(sub) > 1:
            steps = [s.strip() for s in sub if s.strip()]
    
    return steps


def extract_topic(topics_flat: list[str]) -> tuple[str, int]:
    """Extract the primary top-level topic and its ID."""
    if not topics_flat:
        return "Unknown", -1
    
    # Get top-level from first topic
    top = topics_flat[0].split(" > ")[0]
    topic_id = TOPIC_MAP.get(top, -1)
    return top, topic_id


def check_answer_in_solution(solution: str, final_answer: Optional[str]) -> bool:
    """
    Check if the solution text contains/reaches the expected final answer.
    Simple heuristic: does the answer string appear in the last 30% of the solution?
    """
    if not final_answer or not solution:
        return False
    
    # Normalize both
    ans_norm = final_answer.strip().lower()
    sol_tail = solution[int(len(solution) * 0.7):].lower()
    
    # Direct containment
    if ans_norm in sol_tail:
        return True
    
    # Try without LaTeX
    ans_clean = re.sub(r'[\\${}]', '', ans_norm)
    sol_clean = re.sub(r'[\\${}]', '', sol_tail)
    if ans_clean and ans_clean in sol_clean:
        return True
    
    return False


def format_structured_sample(row: dict, corrupt_prob: float = 0.1) -> str:
    """
    Format one MathNet row into structured training format with verification tokens.
    
    Format:
        <topic> TopicName
        <|problem|>
        problem text
        <|solution|>
        <step> Step 1 text <verify> <correct>
        <step> Step 2 text <verify> <correct>
        ...
        <answer> final_answer
        <|end|>
    """
    prob = row.get("problem_markdown", "") or ""
    sols = row.get("solutions_markdown", None) or []
    sol = sols[0] if sols else ""
    topics = row.get("topics_flat", None) or []
    final_answer = row.get("final_answer", None)
    
    # Topic
    top_topic, topic_id = extract_topic(topics)
    
    # Parse solution into steps
    steps = parse_steps(sol)
    
    # Build structured output
    parts = []
    
    # Topic header
    if topics:
        parts.append(f"<topic> {top_topic}")
    
    # Problem
    parts.append(f"<|problem|>\n{prob}")
    
    # Solution with step-level verification
    parts.append("<|solution|>")
    
    if steps:
        # Check if solution reaches correct answer
        has_correct_answer = check_answer_in_solution(sol, final_answer)
        
        for i, step in enumerate(steps):
            is_last = (i == len(steps) - 1)
            parts.append(f"<step> {step}")
            
            # Verification signal
            if has_correct_answer or final_answer is None:
                # Solution is correct (or we can't verify) — all steps correct
                parts.append("<verify> <correct>")
            else:
                # Answer not found in solution — mark last step as needing revision
                if is_last:
                    parts.append("<verify> <revise>")
                else:
                    parts.append("<verify> <correct>")
    else:
        # No parseable solution
        if sol.strip():
            parts.append(f"<step> {sol.strip()}")
            parts.append("<verify> <correct>")
    
    # Final answer
    if final_answer:
        parts.append(f"<answer> {final_answer}")
    
    parts.append("<|end|>")
    
    return "\n".join(parts)


def prepare_structured():
    """Prepare structured training data."""
    os.makedirs(STRUCTURED_DIR, exist_ok=True)
    
    train_path = os.path.join(STRUCTURED_DIR, "train.bin")
    val_path = os.path.join(STRUCTURED_DIR, "val.bin")
    meta_path = os.path.join(STRUCTURED_DIR, "meta.txt")
    
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Structured data already prepared in {STRUCTURED_DIR}")
        return
    
    print("Downloading MathNet dataset (streaming) ...")
    from datasets import load_dataset
    ds = load_dataset("ShadenA/MathNet", split="train", streaming=True)
    
    tok = Tokenizer()
    print(f"Tokenizer: {tok._kind}  vocab_size={tok._vocab_size}")
    
    all_tokens = []
    n_samples = 0
    n_bytes = 0
    n_steps_total = 0
    n_with_answer = 0
    n_verified = 0
    topic_counts = {}
    
    for row in ds:
        text = format_structured_sample(row)
        n_bytes += len(text.encode("utf-8"))
        
        # Tokenize — special tokens are encoded as their text representation
        # The model will learn to predict these as subword tokens
        tokens = tok.encode(text)
        all_tokens.extend(tokens)
        
        # Stats
        n_samples += 1
        steps = parse_steps((row.get("solutions_markdown") or [""])[0] if row.get("solutions_markdown") else "")
        n_steps_total += len(steps)
        if row.get("final_answer"):
            n_with_answer += 1
            sol = (row.get("solutions_markdown") or [""])[0] if row.get("solutions_markdown") else ""
            if check_answer_in_solution(sol, row["final_answer"]):
                n_verified += 1
        
        top, _ = extract_topic(row.get("topics_flat", []))
        topic_counts[top] = topic_counts.get(top, 0) + 1
        
        if n_samples % 5000 == 0:
            print(f"  {n_samples} samples  |  {len(all_tokens):,} tokens  |  {n_steps_total:,} steps")
    
    print(f"Done: {n_samples} samples  |  {len(all_tokens):,} tokens")
    print(f"  Steps total: {n_steps_total:,}  |  avg: {n_steps_total/n_samples:.1f} steps/problem")
    print(f"  With answer: {n_with_answer}  |  Answer verified in solution: {n_verified}")
    print(f"  Topics: {topic_counts}")
    
    bpt = n_bytes / len(all_tokens) if all_tokens else 1.0
    
    arr = np.array(all_tokens, dtype=np.uint16)
    n_val = int(len(arr) * VAL_RATIO)
    val_arr = arr[-n_val:]
    train_arr = arr[:-n_val]
    
    train_arr.tofile(train_path)
    val_arr.tofile(val_path)
    
    with open(meta_path, "w") as f:
        f.write(f"bytes_per_token={bpt:.6f}\n")
        f.write(f"n_samples={n_samples}\n")
        f.write(f"total_tokens={len(arr)}\n")
        f.write(f"n_steps={n_steps_total}\n")
        f.write(f"avg_steps_per_problem={n_steps_total/n_samples:.2f}\n")
        f.write(f"n_with_answer={n_with_answer}\n")
        f.write(f"n_answer_verified={n_verified}\n")
    
    print(f"Saved  train: {len(train_arr):,} tokens → {train_path}")
    print(f"Saved  val:   {len(val_arr):,} tokens → {val_path}")


def show_stats():
    """Show examples of structured formatting."""
    from datasets import load_dataset
    ds = load_dataset("ShadenA/MathNet", split="train", streaming=True)
    
    print("=== STRUCTURED FORMAT EXAMPLES ===\n")
    for i, row in enumerate(ds):
        if i >= 5:
            break
        
        text = format_structured_sample(row)
        steps = parse_steps((row.get("solutions_markdown") or [""])[0] if row.get("solutions_markdown") else "")
        
        print(f"--- Example {i+1} ({row.get('problem_type')}, {len(steps)} steps) ---")
        # Show first 800 chars
        print(text[:800])
        if len(text) > 800:
            print(f"  ... ({len(text)} chars total)")
        print()


if __name__ == "__main__":
    import sys
    if "--stats" in sys.argv:
        show_stats()
    else:
        prepare_structured()
