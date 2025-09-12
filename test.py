import sys
sys.path.append("C:/gitrepos/cloudGPT")

from model import LMSteinshark
from data import load_tokenizer
import json
import time

# ----------------------------
# CONFIG
# ----------------------------
CUR_MODEL       = "D:/nlp/models/PretrainLMSteinshark"
CUR_TOK         = "C:/gitrepos/cloudgpt/tokenizer"

N_TOKENS   = 100
TEMPERATURE = 1.0
TOP_P       = 0.9
TOP_K       = 0
P_MODE      = True  # True=top_p, False=top_k

# ----------------------------
# LOAD MODEL & TOKENIZER
# ----------------------------
MODEL     = LMSteinshark.from_loadpoint(CUR_MODEL, p_override=0).bfloat16().cuda().eval()
TOKENIZER = load_tokenizer(CUR_TOK)

# ----------------------------
# TEST PROMPTS
# ----------------------------
PROMPTS = {
    "Python coding": [
        "Write a Python function to compute Fibonacci numbers:\n\ndef fib(n):",
        "Import numpy and create a 3x3 identity matrix:"
    ],
    "Mathematics": [
        "Solve for x: 2x + 5 = 13",
        "Compute the derivative of f(x) = x**2 + 3*x - 7"
    ],
    "Physics": [
        "State Ohm's law",
        "Explain Newton's second law of motion"
    ],
    "General knowledge": [
        "What is the capital of France?",
        "Who wrote 'Pride and Prejudice'?"
    ],
    "Reasoning": [
        "If all humans are mortal, and Socrates is a human, is Socrates mortal?",
        "Alice is taller than Bob. Bob is taller than Charlie. Who is the shortest?"
    ]
}

# ----------------------------
# GENERATION FUNCTION
# ----------------------------
def generate_completion(prompt, n_tokens=N_TOKENS, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, p_mode=P_MODE):
    tokens_collected = []

    # Use your interface
    for token in MODEL.token_streamer(
        prompt,
        TOKENIZER,
        n_tokens=n_tokens,
        temperature=temperature,
        topk=top_k,
        topp=top_p,
        mode='p' if p_mode else 'k',
        verbose=False
    ):
        tokens_collected.append(token)

    completion_text = "".join(tokens_collected)
    return completion_text, len(tokens_collected)

# ----------------------------
# RUN EVALUATION
# ----------------------------
results = []

for category, prompts in PROMPTS.items():
    for prompt in prompts:
        start_time = time.time()
        completion, token_count = generate_completion(prompt)
        elapsed = time.time() - start_time

        results.append({
            "category": category,
            "prompt": prompt,
            "completion": completion,
            "tokens_generated": token_count,
            "time_sec": elapsed
        })

# ----------------------------
# REPORT & SAVE RESULTS
# ----------------------------
for r in results:
    print(f"[{r['category']}] Prompt: {r['prompt']}")
    print(f"Completion ({r['tokens_generated']} tokens, {r['time_sec']:.2f}s):\n{r['completion']}\n")
    print("-"*80)

# Save JSON for analysis
with open("base_model_eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
