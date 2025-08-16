from datasets import load_dataset
import html
import re
import json 
import os 

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
    
    if question and (len("".join(long_answers)) > 128) and (len("".join(short_answers)) > 1):
        return (question,long_answers,short_answers)
    
    return None
   

def build_nq(split="train"):
    # Load the simplified train set
    dataset         = load_dataset("natural_questions","default",split=split,streaming=True)
    small_dataset   = []
    saved_on        = set()
    for i, example in enumerate(dataset):
        p_ex    = process_q(example)
        if p_ex is not None:
            small_dataset.append(p_ex)

        #Save every 2000 samples
        n_ex        = len(small_dataset) + 1 
        if (not n_ex in saved_on) and (len(small_dataset)+1) % 2000 == 0:
            with open("finetune/nq.json",'w',encoding='utf_8') as wf:
                wf.write(json.dumps(small_dataset))
            print(f"wrote {len(small_dataset)} samples")
            saved_on.add(n_ex)


    return small_dataset

def build_eli5():
    ds = load_dataset("sentence-transformers/eli5",split='train',streaming=True)

    small_dataset   = [] 
    for i,data in enumerate(ds):    
        small_dataset.append((data['question'],data['answer']))

    with open("finetune/eli5.json",'w',encoding='utf_8') as wf:
        wf.write(json.dumps(small_dataset))
        print(f"wrote {len(small_dataset)} samples")

def build_trivia():
    dataset = load_dataset("sentence-transformers/trivia-qa", "pair", split="train",streaming=True)

    for i,data in enumerate(dataset):
        input(f"\n\n{data['query']}\n\n{data['answer']}")

def build_dolly():
    ds              = load_dataset("databricks/databricks-dolly-15k",split='train',streaming=True)
    small_dataset   = [] 
    for i,data in enumerate(ds):
        
        small_dataset.append((data['instruction'],data['response']))
    
    with open("finetune/dolly.json",'w',encoding='utf_8') as wf:
        wf.write(json.dumps(small_dataset))
        print(f"wrote {len(small_dataset)} samples")
    
def build_baize():
    ds = load_dataset("taskydata/baize_chatbot",split='train',streaming=True)
    small_dataset = [] 

    for i,data in enumerate(ds):
        text:str    = data['input']
        text        = text.split('[|Human|]',maxsplit=1)[1]
        question    = text.split('[|AI|]')[0].split()
        answer      = text.split('[|AI|]')[1].split('[|Human|]')[0].split()

        small_dataset.append((question,answer))

    with open("finetune/baize.json",'w',encoding='utf_8') as wf:
        wf.write(json.dumps(small_dataset))
        print(f"wrote {len(small_dataset)} samples")

def compile_dataset():

    full_dataset    = []
    for source in ["baize.json","dolly.json","eli5.json","nq.json"]:
        with open(os.path.join("finetune",source),'r',encoding='utf_8') as rf:
            samples     = json.loads(rf.read())
        
        for data in samples:
            data        = [" ".join(data[i]) for i in range(len(data))]  
            full_dataset.append(data)

    with open(f"finetune/finetune1.json",'w',encoding='utf_8') as wf:
        wf.write(json.dumps(full_dataset))

    print(f"generated {len(full_dataset)} training examples")


def load_nq():

    with open("finetune/nq.json",'r',encoding='utf_8') as rf:
        small_dataset   = json.loads(rf.read())

        for item in small_dataset:

            q,l,s = item 

            print(f"\n\n\n\n\n{q}\n\n{l}\n\n{s}")
            input('')


if __name__ == "__main__":  
    #data = build_nq("train")
    #build_dolly()
    #build_baize()
    #build_eli5()
    #build_trivia()
    #load_nq()
    compile_dataset()
