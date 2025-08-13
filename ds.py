from datasets import load_dataset
import html
import re
import json 

def strip_html(text):
    # Unescape HTML entities
    text = html.unescape(text)
    # Remove HTML tags
    return re.sub(r"<[^>]*>", "", text)

def process_q(entry:dict):

    question        = entry['question']['text'].strip()
    token_list      = entry['document']['tokens']['token']
    long_answers    = [strip_html(" ".join(token_list[poss['start_token']:poss['end_token']])) for poss in entry['annotations']['long_answer']]
    short_answers   = list(set([poss['text'][0] for poss in entry['annotations']['short_answers'] if poss['text']]))
    
    if question and (len("".join(long_answers)) > 128) and len("".join(short_answers) > 4):
        return (question,long_answers,short_answers)
    
    return None
    # Extract the first long answer (if present)
    answer = ""
    for ann in answer_list:
        if ann.get("long_answer", {}).get("start_token", -1) != -1:
            answer = ann["long_answer"].get("text", "")
            break
    
    # If no long answer found, fallback to short answer list
    if not answer:
        for ann in answer_list:
            if "short_answers" in ann and ann["short_answers"]:
                answer = " ".join(sa.get("text", "") for sa in ann["short_answers"])
                break

    # Clean up
    question = strip_html(question)
    answer = strip_html(answer)

    if question and answer:
        return ((question, answer))
    else:
        return None

def format_nq_simplified(split="train"):
    # Load the simplified train set
    dataset = load_dataset("natural_questions","dev",split=split,streaming=True)
    small_dataset = []
    

    for i, example in enumerate(dataset):
        p_ex    = process_q(example)
        if p_ex is not None:
            small_dataset.append(p_ex)

        #Save every 2000 samples
        if (i+1) % 2000 == 0:
            with open("finetune/nq.json",'w',encoding='utf_8') as wf:
                wf.write(json.dumps(small_dataset))
    
    

    

    return small_dataset

if __name__ == "__main__":
    data = format_nq_simplified("validation")
    print(f"Sample: {data[0]}")
