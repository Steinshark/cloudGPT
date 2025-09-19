from datasets import load_dataset
import html
import re
import json 
import os 
import random
import sys 
sys.path.append("C:/gitrepos/cloudGPT")
from utils import END_TOKEN,           CODE_TOKEN,          RUNCODE_TOKEN,       WEB_TOKEN,           PROMPT_TOKEN,        RESPONSE_TOKEN,      RESERVE_1,           RESERVE_2



instruct_variants   = [
        "Here is a task for you to solve:",
        "Please complete the following request:",
        "Given the instruction below, provide the correct output:",
        "Your task is:",
        "Answer the following question:",
        "Solve the task described here:",
        "Follow the instruction and provide a response:",
        "Complete the request that follows:",
        "Respond appropriately to the task below:",
        "Carry out the instruction provided:",
        "Read the instruction and answer accordingly:",
        "Provide a solution for the following task:",
        "The following is a task. Please respond:",
        "Execute the instruction given below:",
        "Based on the task described, give a proper response:",
        "Follow the directions and complete the request:",
        "Here is an instruction. Respond accordingly:",
        "Process the following instruction and answer:",
        "Below is a request. Please provide the response:",
        "Complete the following instruction appropriately:",
        "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        ]


# Variants for input prompts
input_variants = [
        "Here is the provided input:",
        "Use this input:",
        "Base it off of this input:",
        "Utilize the following input for your code:",
        "Here is some input to use:",
        "The following input should be processed:",
        "Given these values:",
        "Use the data below:",
        "Consider the following input:",
        "here is some input to use",
        'this is some input to use',
        'the code should use this',
        'Base the code off of this input'
        "### Input:"  # occasional structured header
        ]

def expand_prompts(prompt_completion_pair:list[tuple[str,str]]):
    completions     = []

    if isinstance(prompt_completion_pair[0],str):
        return [(prompt_completion_pair[0],prompt_completion_pair[1])]
    
    for prompt in prompt_completion_pair[0]:
        for completion in prompt_completion_pair[1]:
            completions.append((prompt,completion))
    
    return completions

def read_data():
    content     = open("C:/gitrepos/cloudGPT/finetune/training1.txt",'r',encoding='utf-8').read()
    dataset     = eval(content)
    import pprint 
    data = []
    for item in dataset:
        prompts = expand_prompts(item)
        data += prompts 
    #codeex      = [ex for ex in data if RUNCODE_TOKEN in ex[1]]
    import pprint 
    for d in data:

        input(d)

    prompts     = [f"{PROMPT_TOKEN}{it[0]}{RESPONSE_TOKEN}{it[1]}" for it in data]    
    
    with open(f"C:/gitrepos/cloudgpt/finetune/runcode.jsonl",'w',encoding='utf-8') as writefile:
        for item in prompts:
            writefile.write(f"{json.dumps(item)}\n")
    
    print(f"built {len(prompts)} examples")

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
   

#Simplified NQ -> much better version to be had
def build_nq(split="train"):
    # Load the simplified train set
    dataset         = load_dataset("florin-hf/nq_open_gold",streaming=True)['train']
    small_dataset   = []
    for i, example in enumerate(dataset):

        
        question    = example['question']
        answer      = example['text']

        small_dataset.append(f"{PROMPT_TOKEN}{question}{RESPONSE_TOKEN}{answer}")

    with open("finetune/nq.json",'w',encoding='utf_8') as wf:
        wf.write(json.dumps(small_dataset))
    print(f"wrote {len(small_dataset)} samples")


    return small_dataset

#ELI5 is always consistent
def build_eli5():
    ds = load_dataset("sentence-transformers/eli5",split='train',streaming=True)

    small_dataset   = [] 
    for i,data in enumerate(ds):    
        small_dataset.append(f"{PROMPT_TOKEN}{data['question']}{RESPONSE_TOKEN}{data['answer']}")

    with open("finetune/eli5.json",'w',encoding='utf_8') as wf:
        wf.write(json.dumps(small_dataset))
        print(f"wrote {len(small_dataset)} samples")

#This one sucks!
def build_trivia():
    dataset = load_dataset("sentence-transformers/trivia-qa", "pair", split="train",streaming=True)

    for i,data in enumerate(dataset):
        input(f"\n\n{data['query']}\n\n{data['answer']}")

#This one is not great!
def build_dolly():
    ds              = load_dataset("databricks/databricks-dolly-15k",split='train',streaming=True)
    small_dataset   = [] 
    for i,data in enumerate(ds):
        
        small_dataset.append((data['instruction'],data['response']))
    
    with open("finetune/dolly.json",'w',encoding='utf_8') as wf:
        wf.write(json.dumps(small_dataset))
        print(f"wrote {len(small_dataset)} samples")

#Excellent for instruct!
def build_baize():
    ds = load_dataset("linkanjarad/baize-chat-data",split='train',streaming=True)
    small_dataset = [] 

    for i,data in enumerate(ds):
        text:str    = data['chat_sample']
        HUMAN       = '[|Human|]'
        AI          = '[|AI|]'

        text        = text.split(HUMAN,maxsplit=1)[1]
        text        = text.replace(HUMAN,PROMPT_TOKEN).replace(AI,RESPONSE_TOKEN)
        small_dataset.append(f"{PROMPT_TOKEN}{text}")

    with open("finetune/baize.json",'w',encoding='utf_8') as wf:
        wf.write(json.dumps(small_dataset))
        print(f"wrote {len(small_dataset)} samples")

#Great dataset
def build_reddit():
    path = r"D:\nlp\finetune\reddit.txt"
    small_dataset = [] 
    text  = open(path,'r',encoding='utf-8').read()

    conversations   = text.split("<|endoftext|>")

    for conv in conversations:
        try:
            q,a = conv.split('\n\n')[:2]
            small_dataset.append(f"{PROMPT_TOKEN}{q}{RESPONSE_TOKEN}{a}")
        except ValueError:
            pass

        #input(f'{a} -> {q}')
    with open("finetune/reddit.json",'w',encoding='utf_8') as wf:
        wf.write(json.dumps(small_dataset))
        print(f"wrote {len(small_dataset)} samples")

#Good coding 
def build_code_alpaca():
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca",split='train')



    small_dataset = [] 

    for i,data in enumerate(ds):
        prompt      = data['prompt'].replace('Below is an instruction that describes a task. Write a response that appropriately completes the request.',random.choice(instruct_variants))
        #Replace input
        prompt      = prompt.replace("### Instruction:\n",'')
        prompt      = prompt.replace('### Input:',random.choice(input_variants))

        prompt      = random.choice(instruct_variants) + '\n' + data['instruction'] + '\n' + random.choice(input_variants) + "\n"
        prompt      = prompt + data['input']

        example     = f"{PROMPT_TOKEN}{prompt}{RESPONSE_TOKEN}{data['output']}"

        small_dataset.append(f"{example}")

    with open("finetune/code_alpaca.json",'w',encoding='utf_8') as wf:
        wf.write(json.dumps(small_dataset))
        print(f"wrote {len(small_dataset)} samples")

#Add all currently used datasets into "finetune1.json"
def compile_dataset():

    full_dataset    = []
    sources         = ["baize.json","eli5.json","nq.json",'reddit.json']
    sources         = ["ELI5_fixed.json","nq.json",'code_alpaca.json',"runcode.jsonl"]
    for source in sources:
        with open(os.path.join("finetune",source),'r',encoding='utf_8') as rf:
            if 'jsonl' in source:
                samples     = []
                for line in rf:
                    samples.append(json.loads(line))
            else:
                samples     = json.loads(rf.read())
        
        for data in samples:
            full_dataset.append(data)

    with open(f"finetune/finetune.json",'w',encoding='utf_8') as wf:
        wf.write(json.dumps(full_dataset))

    print(f"generated {len(full_dataset)} training examples")


def load_nq():

    with open("finetune/nq.json",'r',encoding='utf_8') as rf:
        small_dataset   = json.loads(rf.read())

        for item in small_dataset:

            q,l,s = item 

            print(f"\n\n\n\n\n{q}\n\n{l}\n\n{s}")
            input('')


def fix_eli():
    pattern     = re.compile(r'(\[([A-Za-z0-9]+)\]\(_URL_(\d+)_\))')
    data = json.loads(open("C:/gitrepos/cloudGPT/finetune/eli5.json",'r',encoding="utf-8").read())

    newdata      = []
    for d in data:
        res     = pattern.findall(d)
        if res:
            original = res[0][0]
            new     = res[0][1]
            newdata.append(d.replace(original,new))
        else:
            newdata.append(d)
    
    with open("finetune/ELI5_fixed.json",'w',encoding='utf-8') as writefile:
        writefile.write(json.dumps(newdata))
    

if __name__ == "__main__":  
    #fix_eli()
    #build_code_alpaca()
    #read_data()
    #build_nq()
    #build_baize()
    #build_reddit()
    #build_eli5()
    #read_data()
    compile_dataset()

    #data = build_nq("train")
    #build_dolly()
    #build_baize()
    #build_eli5()
    #build_trivia()
    #load_nq()
    #compile_dataset()
