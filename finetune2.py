from model import LMSteinshark
from data import load_tokenizer
from torch.utils.data import Dataset, DataLoader
from utils import END_TOKEN,PROMPT_TOKEN,RESPONSE_TOKEN,SPECIAL_TOKENS,RESERVE_1
from environment import MAX_NORM
from tokenizers.implementations import ByteLevelBPETokenizer as BPET
import random 
import torch 
import numpy 
import os 
import json 
import itertools

#Finetune data will always be in a single .txt file 
#This class will tokenizer it and chunk it into 2048 sized chunks separated by <|endoftext|>
class pretrain_ds(Dataset):

    def __init__(self,trainset:list[list[int]|list[numpy.int32]],tokenizer:BPET,context=2048):
        
        self.data               = trainset

        self.context            = context

        self.n_tokens           = len(list(itertools.chain(*self.data)))
        
        print(f"loaded dataset of {self.n_tokens//1_000_000}M tokens -> {len(self.data)} sequences")

    #Just return any random sequence starting with an endoftext token
    def __getitem__(self,i:int):

        #Pick a random start sequence
        seq                 = random.choice(self.data)


        sample_tokens       = torch.tensor(seq[:-1])
        sample_targets      = torch.tensor(seq[1:])


        return sample_tokens,sample_targets

    def __len__(self):
        return len(self.data)

def expand_prompts(prompt_completion_pair:list[tuple[str,str]]):
    completions     = []

    if isinstance(prompt_completion_pair[0],str):
        return [(prompt_completion_pair[0],prompt_completion_pair[1])]
    
    for prompt in prompt_completion_pair[0]:
        for completion in prompt_completion_pair[1]:
            completions.append((prompt,completion))
    
    return completions

def build_ds():
    writer = open("C:/gitrepos/cloudGPT/finetune/unsupervised1.txt",'w',encoding='utf_8')

    all_resp    = []
    finetune_data = eval(open("C:/gitrepos/cloudGPT/finetune/training1.txt",'r',encoding='utf_8').read())
    for resp in finetune_data:
        all_resp += expand_prompts(resp)
    
    writer.write(json.dumps(all_resp))
    #writer.write('[\n')

    # for a,b in all_resp:
    #     writer.write(f'({a},{b}),\n')
    # print(all_resp[0])

    #print(f"generated {len(all_resp)} pairs -> {len(''.join([a+b for a,b in all_resp]))}")
    writer.close() 

def train_on_reddit():

    lm_model                            = LMSteinshark.from_loadpoint("D:/production",p_override=0).cuda().bfloat16()
    lm_model.name                       = "finetune1"
    optimizer                           = torch.optim.AdamW(lm_model.parameters(),lr=2e-4)
    tokenizer                           = load_tokenizer("C:/gitrepos/cloudgpt/tokenizer")
    tokens                              = generate_finetune_prompts() 
    ds                                  = pretrain_ds(tokens,tokenizer)

    bs                                  = 1
    input_size                          = 2048
    vocab_size                          = 32768
    accu_steps                          = 128
    dataloader                          = DataLoader(ds,batch_size=bs,shuffle=True)
    accu_loss                           = 0 
    print(f"begin train",flush=True)

    for ep in range(8):

        for i,batch in enumerate(dataloader):
            input_ids                   = batch[0].long().cuda()
            target_ids                  = batch[1].long().cuda()

            seq_len                     = input_ids.size(-1)

            logits,target_ids           = lm_model.forward(input_ids,target_ids)
            logits                      = logits.view(bs*seq_len,vocab_size)
            targets                     = target_ids.view(bs*seq_len)

            #Compute loss and backprop
            loss:torch.Tensor           = torch.nn.functional.cross_entropy(logits, targets) / accu_steps
            accu_loss += loss
            loss.backward()

            if ((i % accu_steps) == 0) and (not i == 0):
                torch.nn.utils.clip_grad_norm_(lm_model.parameters(),MAX_NORM)
                optimizer.step()
                optimizer.zero_grad()
                print(f"\tloss [{ep} | {i}/{len(dataloader)}] = {accu_loss.detach().cpu().float()}")
                accu_loss               = 0 

        lm_model.save("C:/gitrepos/cloudgpt/models/",save_weights=True)
        print(f"weights saved as {lm_model.name}")

def generate_finetune_prompts():

    build_ds()
    #Path is finetune/unsupervised1.txt
    fpath                               = "finetune/unsupervised1.txt"
    data                                = json.loads(open(fpath,'r',encoding='utf_8').read())
    tokenizer                           = load_tokenizer('tokenizer')

    assert len(tokenizer.encode("".join(SPECIAL_TOKENS)).ids) == len(SPECIAL_TOKENS)

    finetunedata                        = []
    samples                             = [] 
    n_tokens                            = 0 
    for prompt,response in data:
        sample                          = tokenizer.encode(f"{PROMPT_TOKEN}{prompt}{PROMPT_TOKEN}{RESPONSE_TOKEN}{response}{RESPONSE_TOKEN}{RESERVE_1}").ids
        finetunedata.extend(sample)
        samples.append(sample[:-1])

    numpy.save("finetune/unsupervised",numpy.asarray(finetunedata,dtype=numpy.int32))
    print(f"generated {len(finetunedata)//1_000}K tokens")

    #return a stream of tokens 
    return samples

def create_responses():
    model           = LMSteinshark.from_loadpoint("config_path",p_override=0).bfloat16().cuda().eval()
    tokenizer       = load_tokenizer("C:/gitrepos/cloudgpt/tokenizer")

    prompts         = open("finetune/rlhf_promtps.txt",'r',encoding='utf_8').readlines()

    data            = {prompt:[] for prompt in prompts}

    for prompt in data:

        for _ in range(10):
            token_ids   = tokenizer.encode(prompt).ids 

            response    = ""
            for tok in model.token_streamer(prompt,tokenizer,2048,.6,topk=150,topp=.5):
                response += tok 
            
            data[prompt].append(response)
            




if __name__ == '__main__':

    #scp ubuntu@192.222.58.74:~/Stein2/cloudgpt/models/production/* D:/production
    generate_finetune_prompts()
    train_on_reddit()