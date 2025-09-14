from model import LMSteinshark
from data import *
from torch.utils.data import Dataset, DataLoader
from utils import END_TOKEN,PROMPT_TOKEN,RESPONSE_TOKEN,SPECIAL_TOKENS,RESERVE_1
from environment import MAX_NORM, ENV_PREFIX
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




if __name__ == '__main__':

    LR                  = 2e-5
    WD                  = 1e-4
    EP                  = 3
    BS                  = 16
    ACCU                = (1024*1024) // 2048
    SAVE                = 16
    STEP_EVERY          = ACCU // BS
    CONTEXT             = 1024

    #Load tokenizer
    fpath_tok           = f"{ENV_PREFIX}/tokenizer"
    tokenizer           = load_tokenizer(fpath_tok)
    ftt                 = FinetuneTokenizer(tokenizer,CONTEXT,RESERVE_1)
    VS                  = ftt.base_tokenizer.get_vocab_size()

    #Load data
    fname               = f'{ENV_PREFIX}/data/factual_dataset_select.jsonl'
    dataset             = FinetuneDataset(fname, ftt,data_cap=32*1024)
    loader              = DataLoader(dataset,batch_size=BS,shuffle=True)

    #Load model
    model_loadpoint     = f"{ENV_PREFIX}/models/PreTrainLMSteinshark"
    lm_model            = LMSteinshark.from_loadpoint(model_loadpoint,p_override=.1).bfloat16().cuda()
    lm_model.name       = "FactTune"

    #Build optimizer
    optim               = torch.optim.AdamW(lm_model.parameters(),lr=LR,weight_decay=WD)
    save_count          = 0
    accumulation_loss   = 0
    
    #Train
    for p in lm_model.parameters():
        p.requires_grad_(True)

    for ep in range(EP):
        c_loss  = 0 

        for i,batch in enumerate(loader):

            #Snatch the data
            input_ids           = batch[0].cuda()
            target_ids          = batch[1].cuda()
            attn_mask           = batch[2].cuda()
            #input(f"train on {tokenizer.decode(input_ids[0][:16].cpu().numpy())}")
            #input(f"train on {tokenizer.decode(target_ids[0][:16].cpu().numpy())}")
            #Send it forward
            #input(f"Shapes:\n{input_ids.shape}\n{target_ids.shape}\n{attn_mask.shape}")
            logits,target_ids   = lm_model.forward(input_ids,target_ids,attn_mask)

            #Shape it up
            logits              = logits.view(target_ids.size(0)*target_ids.size(1),VS)
            targets             = target_ids.view(target_ids.size(0)*target_ids.size(1))
            
            #Calculate loss -> grads will be divided by STEP_EVERY once right before stepping
            loss                = torch.nn.functional.cross_entropy(logits,targets,ignore_index=ftt.pad_token_id)
            accumulation_loss   += loss.item()
            loss.backward()

            if ((i + 1) % (STEP_EVERY)) == 0:
                for p in lm_model.parameters():
                    if p.grad is not None:
                        p.grad.div_(STEP_EVERY)

                #Clip norms and step + reset
                torch.nn.utils.clip_grad_norm_(lm_model.parameters(),MAX_NORM)
                optim.step()
                optim.zero_grad()
                accumulation_loss /= STEP_EVERY
                print(f"EP {ep}\t[{i+1}\t/{len(loader)}] -\t{accumulation_loss}")
                c_loss          = 0 

                if (save_count % SAVE) == 0:
                    lm_model.stats['losses'].append(accumulation_loss)
                    lm_model.name = f"PreFinetune{save_count}"
                    lm_model.save(save_weights=True,root=f'{ENV_PREFIX}/models')
                    print(f"\tsaving {lm_model.name}")
                
                save_count += 1 
                torch.cuda.empty_cache()

            tot_tok_thru    = attn_mask.int().sum().item()

            #Update stats 
            lm_model.stats['run_tok_through'] += tot_tok_thru
            lm_model.stats['run_iter_through'] += 1

            lm_model.stats['tok_through'] += tot_tok_thru
            lm_model.stats['iter_through'] += 1

            
        print(f"\n\nEP {ep} complete\n\n")
        lm_model.save(save_weights=True,root=f'{ENV_PREFIX}/models')
