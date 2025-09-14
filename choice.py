import json
import os
import keyboard  # pip install keyboard
from classify import classify, load_model
import random

INPUT_FILE  = r"//Steinpc/s/nlp/data/factual_dataset.jsonl"
OUTPUT_FILE = r"//Steinpc/s/nlp/data/relevant_topics.jsonl"

SAVE_INTERVAL = 10
import re

def filter_line(line:str):

    if "Category:" in line:
        return False 
    
    if 'thumb|' in line:
        return False 
    
    if len(line.split('<|prompt|>')[1]) < 120:
        return False  
    
    return True 


def match_keys(title:str):
    return title.lower().replace(" ","").replace("_","")

prompt_templates = [
    "Provide factual information about {subject}.",
    "What are the key details about {subject}?",
    "Summarize important facts regarding {subject}.",
    "Give a concise explanation of {subject}.",
    "Describe the main points about {subject}.",
    "What is known about {subject}?",
    "Provide an overview of {subject}.",
    "List the relevant facts related to {subject}.",
    "Explain the significance of {subject}.",
    "Give a factual summary of {subject}.",
    "What are the notable aspects of {subject}?",
    "Share accurate information concerning {subject}.",
    "What details are known about {subject}?",
    "Provide a short factual description of {subject}.",
    "Summarize the topic {subject} in a factual manner.",
    "What does research say about {subject}?",
    "Present key points about {subject}.",
    "Offer an objective overview of {subject}.",
    "Explain {subject} in a factual, unbiased way.",
    "Give a clear and accurate summary of {subject}.",
    "Tell me more about {subject}",
    "Talk about {subject}",
    "What is {subject}",
    "Talk about {subject}",
    "What are some facts about {subject}?",
    "What can you tell me about {subject}?",
    "{subject}",
    "{subject} - talk about this.",
    "I'm curious about {subject}",
    "I want to know about {subject}.",
    "Tell me about {subject}",
    "Give some info on {subject}"
]

processed_items = []
# Compile regexes once
regexes         = []

# Sort templates by length descending, so more specific match first
prompt_templates.sort(key=len, reverse=True)
removers        = [] 
training_set    = [] 
score_min       = 80
for t in prompt_templates:
    # Split by {subject}, escape everything else
    parts = t.split("{subject}")
    for p in parts:
        removers.append(p)
    continue
    pattern = "".join([re.escape(p) for p in parts])
    # Add a non-greedy capture group for {subject}
    pattern = pattern.replace("", "(.+?)")
    # Anchor and allow optional leading/trailing whitespace
    compiled = re.compile(r"^\s*" + pattern.strip() + r"\s*$")
    regexes.append(compiled)
    # Print pattern for debugging
    print(compiled.pattern)

removers.sort(key = lambda x:len(x),reverse=True)
def extract_subject(prompt: str) -> str | None:
    for r in removers:
        prompt = prompt.replace(r,'')
    return prompt


def main():

    TRAIN   = False 
    # Ensure output dir exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Figure out how many lines already labeled (resume support)
    processed = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as out_f:
            for _ in out_f:
                processed += 1

    autoskip    = ["Economy of", "Transport in","Telecommunications in ","Politics of","Geography of","History of","Foreign relations",]
    letters     = 'abcdefghijklmnopqrstuvwxyz'
    already     = [json.loads(line)['text'] for line in open(OUTPUT_FILE,'r',encoding='utf_8').readlines()]
    results = []
    counter = 0
    text_per_item = {}
    current_letter = None  # Track first letter of current article
    skip_to_next_letter = False
    data        = []
    print("=== Wikipedia Article Filter ===")
    print("Press ↑ for Relevant, ↓ for Not Relevant, 'b' jump to next letter, Esc to quit.\n")
    print(f"Resuming after {processed} already labeled entries.\n")
    scores      = json.loads(open("//Steinpc/s/nlp/data/article_scores.json",'r',encoding='utf-8').read())

    #if not TRAIN:
       # mod,tok     = load_model()

    #Prev 688MB
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        # Skip lines already processed
        # for _ in range(processed):
        #     next(f)

        for line in f:
            entry = line.strip()
            if not entry:
                continue

            try:
                text = json.loads(entry)
            except json.JSONDecodeError:
                continue

            # Show up to <response>
            display_text    = text.split("<|response|>")[0].replace("<|prompt|>", "")
            article         = extract_subject(display_text).lower()
            article_text    = text.split('<|response|>')[1]

            # if not article or article in processed_items or article in already:
            #     continue
            #processed_items.append(article)

            #if not TRAIN:
                # pred_label, conf    = classify(mod,tok,article)
                # article     = article + '            '
                # input(f"{article[:20]} ->\t{pred_label} ({conf})")

            if match_keys(article) in scores:
                article = match_keys(article)

                if scores[article] > score_min and filter_line(text):
                    data.append(text)
                    #input(text)
                    
                    # if article in text_per_item:
                    #     text_per_item[article] += len(article_text)
                    # else:
                    #     text_per_item[article] = len(article_text)
            else:
                continue

            continue
            # # if random.random() < .0000004:
            # #     break
            # if match_keys(article) in scores:
            #     article = match_keys(article)

            #     if scores[article] > score_min:
            #         input(f"{article} -> {scores[article]}")
            # continue

            

            first_letter = article[0].lower()

            if not first_letter in letters:
                continue

            if current_letter is None:
                current_letter = first_letter
            
            flag = False
            for item in autoskip:
                if item in article:
                    flag = True 
            if flag:
                continue



            if not TRAIN:
                pred_label, conf    = classify(mod,tok,article)
                article     = article + '            '
                input(f"{article[:20]} ->\t{pred_label} ({conf})")

            if TRAIN:
                print("\n---")
                print(article)
                print("\n↑ Relevant | ↓ Not Relevant | b jump next letter | Esc quit")
                key = keyboard.read_event(suppress=True)
                while key.event_type != keyboard.KEY_DOWN:
                    key = keyboard.read_event(suppress=True)
                if key.name == "up":
                    results.append({"text": article, "label": 1})
                    counter += 1
                elif key.name == "down":
                    results.append({"text": article, "label": 0})
                    counter += 1
                elif key.name == "b":
                    skip_to_next_letter = True
                elif key.name == "esc":
                    break

            # If we're skipping to next letter
            if skip_to_next_letter:
                if len(letters) == 1:
                    break 
                else:
                    letters = letters[1:]
                
                skip_to_next_letter = False

            # Save every SAVE_INTERVAL entries
            if TRAIN:
                if counter % SAVE_INTERVAL == 0 and counter > 0:
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
                        for r in results:
                            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    results = []
                    print(f"✅ Saved {counter + processed} entries so far.")

    #Save data 
    with open('//Steinpc/s/nlp/data/factual_dataset_80.jsonl','w',encoding='utf-8') as writefile:
        for item in data:
            writefile.write(json.dumps(item,ensure_ascii=False) + "\n")
    exit()
    if TRAIN:
        if results:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
                for r in results:
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"✅ Final save: {len(results)} new entries.")

    print("Done.")



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
