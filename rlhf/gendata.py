import json
import sys 
sys.path.append("C:/gitrepos/cloudGPT")
from model import LMSteinshark
from data import load_tokenizer

PROMPTS         = "D:/Project Chat/data/prompts.txt"
DATASET         = "D:/Project Chat/data/prompt_responses.json"
CUR_MODEL       = "C:/data/nlp/models/finetune0"
MODEL           = LMSteinshark.from_loadpoint(CUR_MODEL,p_override=0).bfloat16().cuda().eval()
CUR_TOK         = "C:/gitrepos/cloudgpt/tokenizer"
TOKENIZER       = load_tokenizer(CUR_TOK)

temp            = .75 
p               = .99
n_tok           = 128

# List of prompts you want to seed
prompts = [p for p in open(PROMPTS,'r',encoding='utf_8').readlines() if not p.startswith("#")]

# Number of responses per prompt
responses_per_prompt = 3

dataset = {}

for prompt in prompts[:4]:
    dataset[prompt] = []
    for _ in range(responses_per_prompt):
        response = "".join(MODEL.token_streamer(prompt,TOKENIZER,n_tok,temp,None,p))  # Call your model's API
        dataset[prompt].append(response.strip())
    print(f"Collected {responses_per_prompt} responses for: {prompt}")

# Save to dataset.json
with open(DATASET, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print("Dataset generation complete.")
