import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from model import LMSteinshark
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
from dataset import TokenizedDataset, load_tokenizer
import numpy 
import time 
import matplotlib
matplotlib.use("qt5agg")
from matplotlib import pyplot as plt 
import math
import json 
import tkinter as tk
from utils import reduce_arr
from training import *
from tok import * 

#sys.path.append("C:/gitrepos/steinpy/src")
#from steinpy.utils import reduce_arr 

_UPDATE_EVERY_T                 = int(1*60)
_SAMPLE_EVERY_T                 = 10*60
_LAST_UPDATE_T                  = time.time() 
_LAST_SAMPLE_T                  = time.time() - (3*60)
_SAVE_MODEL_EVERY               = 5000
_N_TOKENS                       = None
MAX_NORM                        = 1000

CUR_STEP                        = 0 
TOT_STEP                        = 0
TOK_THRU                        = 0
MODEL                           = None
TOKENIZER                       = None
LOSS                            = None 
PROMPT                          = "<|endoftext|>"

def print_model_info(current_step,total_step,tok_thruput,losses):
    iters                   = "iter " + f"{current_step}/{total_step}".rjust(11) + "   "
    losses                  = f"{float(sum(losses[-64:])) / float(len(losses[-64:])):.5f}".rjust(8) + "   "
    tok_thru                = f"{tok_thruput/1_000:.1f}k tok/s" + "   "
    toks                    = f"{model.stats['tok_through']/1_000_000:.1f}M tokens"
    lr                      = f"  lr={optimizer.param_groups[0]['lr']}"
    _LAST_UPDATE_T          = time.time()

    print(iters+losses+tok_thru+toks+lr)


def print_model_sample(model:LMSteinshark,prompt:str,tokenizer:ByteLevelBPETokenizer,n_tokens:int,temp:float):
    print(f"\n\nGenerating:")
    print(f"{tokenizer.decode(model.generate(tokenizer.encode(prompt).ids,TOKENIZER,n_tokens=n_tokens,temperature=temp,top_k=50))}\n\n")


def print_update(current_step,total_step,losses,tok_thruput,model:LMSteinshark,prompt:str,tokenizer:ByteLevelBPETokenizer,optimizer:torch.optim.Adam,args):
    global _LAST_UPDATE_T
    global _LAST_SAMPLE_T

    #Check for printint stats 
    if time.time() - _LAST_UPDATE_T > _UPDATE_EVERY_T:

        iters                   = "iter " + f"{current_step}/{total_step}".rjust(11) + "   "
        losses                  = f"{float(sum(losses[-64:])) / float(len(losses[-64:])):.5f}".rjust(8) + "   "
        tok_thru                = f"{tok_thruput/1_000:.1f}k tok/s" + "   "
        toks                    = f"{model.stats['tok_through']/1_000_000:.1f}M tokens"
        lr                      = f"  lr={optimizer.param_groups[0]['lr']}"
        _LAST_UPDATE_T          = time.time()

        print(iters+losses+tok_thru+toks+lr)

    #Check to save model 
    if current_step % _SAVE_MODEL_EVERY == 0 and not current_step == 0:
        model.save(MODELS)

    #Check to sample 
    if time.time() - _LAST_SAMPLE_T > _SAMPLE_EVERY_T:
        print(f"\n\nPrompt: {PROMPT}\n\nModel:",end='')
        print(f"{tokenizer.decode(model.generate(tokenizer.encode(prompt).ids,TOKENIZER,n_tokens=256,temperature=.7,top_k=50))}\n\n")
        _LAST_SAMPLE_T          = time.time()


if __name__ == "__main__":

    #Create a basic tk window to change variables on the fly 
    # and allow sampling/info at any point.
    # this will be printed to cmdline but prompting from
    # the tk window versus a fixed interval.
    root                        = tk.Tk()
    root.title("Model Sampler")
    max_tok_entry:tk.Entry      = None 
    temp_entry:tk.Entry         = None 

    # Prompt line
    prompt_label = tk.Label(root, text="Prompt:")
    prompt_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
    prompt_entry = tk.Entry(root, width=60)
    prompt_entry.grid(row=0, column=1, padx=5, pady=5, columnspan=2)
    generate_button = tk.Button(root, text="Generate", command=lambda : print_model_sample(MODEL,prompt_entry.get(),tokenizer=TOKENIZER,n_tokens=int(max_tok_entry.get()),temp=float(temp_entry.get())))
    generate_button.grid(row=0, column=3, padx=5, pady=5)

    # Temperature and Max Token line
    temp_label = tk.Label(root, text="Temp:")
    temp_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
    temp_entry = tk.Entry(root, width=10)
    temp_entry.insert(0, "1.0")
    temp_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

    max_tok_label = tk.Label(root, text="Max Tok:")
    max_tok_label.grid(row=1, column=2, padx=5, pady=5, sticky="e")
    max_tok_entry = tk.Entry(root, width=10)
    max_tok_entry.insert(0, "64")
    max_tok_entry.grid(row=1, column=3, padx=5, pady=5, sticky="w")

    #Update 
    update_label    = tk.Button(root,text="Update",command=lambda : print_model_info(CUR_STEP,TOT_STEP,TOK_THRU,LOSS))
    update_label.grid(row=2,column=0,padx=5, pady=5, sticky='w')


    #Ensure optimizations 
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    #Handle arguments
    argparser                   = argparse.ArgumentParser()
    argparser.add_argument('--model_dir',default='')
    argparser.add_argument('--model_type',default='base')
    argparser.add_argument('--bs',default='16')
    argparser.add_argument("--n_layers",default='16')
    argparser.add_argument('--bs_tok',default='262144')
    argparser.add_argument('--load_vocab',default='True')
    argparser.add_argument('--train_root',default=PATH)
    argparser.add_argument('--ds_name',default='tokens')
    argparser.add_argument('--tokenizer_name',default='32k')
    argparser.add_argument('--input_size',default='256')
    argparser.add_argument('--model_name',default='newmodel')
    argparser.add_argument('--n_embed',default='1024')
    argparser.add_argument('--head_dim',default='128')
    argparser.add_argument('--n_ff',default='4')
    argparser.add_argument('--load',default='False')
    argparser.add_argument('--max_tok',default='5_000_000_000')

    args                        = argparser.parse_args()


    #Load data 
    max_tokens                  = eval(args.max_tok)
    tokens                      = [] 
    dataset,n_tok_loaded        = load_tokens(args,max_tokens)
    _N_TOKENS                   = dataset.n_tokens

    tokenizer_name              = args.tokenizer_name                           #Tokenizer used
    train_root                  = args.train_root                               #Where all the training data will be found  
    
    tokenizer                   = load_tokenizer(f"{train_root}/{tokenizer_name}")


    #Training/Model Settings 
    #Input Settings 
    context1_size               = 0                                             #Sequence to coarse summarize 
    context2_size               = 0                                             #Sequence to fine summarize
    core_size                   = eval(args.input_size)                         #1:1 sequence 
    lg_size                     = 0                                             #Coarse reduced to this 
    md_size                     = 0                                             #Fine reduced to this 
    tfmr_input_size             = core_size + lg_size + md_size                 #Size of sequence going through transformer stack
    input_size                  = context1_size + context2_size + core_size     #Total number of tokens sampled per train step
    vocab_size                  = tokenizer.get_vocab_size()                    #Vocab Size

    #Model settings 
    n_layers                    = eval(args.n_layers)                           #Transformers stacked 
    n_embed                     = eval(args.n_embed)                            #Dimension of the embedding per token             
    n_heads                     = n_embed//eval(args.head_dim)                  #Number of attn heads          
    n_ff                        = int(n_embed*eval(args.n_ff))                  #Size of the feed forward network 
    act_fn                      = torch.nn.GELU                                 #Used throughout model

    #Training settings
    train_batch_tok             = eval(args.bs_tok)                             #Number of tokens before stepping optimizer 
    bs                          = eval(args.bs)                                 #BS used per train iter (NOT per optimizer update)
    lr                          = .0001                                         #Max LR used in OneCycleLR
    wd                          = .01                                           #WD used throughout
    dropout                     = .2                                            #P used throughout
    virtual_bs                  = train_batch_tok // tfmr_input_size            #Number of iters before stepping Optimizer
    accu_steps                  = virtual_bs // bs                              #Number of steps before stepping optimizer
    pct_start                   = .3                                            #Where peak LR will occur       
    train_iters                 = 2* _N_TOKENS // (bs*tfmr_input_size)             #Total iters used to train
    lr_steps                    = 2* _N_TOKENS // train_batch_tok                  #Total steps (used for OneCycleLR)
    tokenizer_name              = args.tokenizer_name                           #Tokenizer used

    #Sampling 
    sample_text                 = "scientists have discovered a new technique for creating large language models"
    PROMPT                      = sample_text
    #Create Tokenizer
    tokenizer               = load_tokenizer(f"{train_root}/{tokenizer_name}")
    assert tokenizer.get_vocab_size() == vocab_size

    #Create model 
    if args.model_type == "summarizer": 
        model:MiniTransformerSteinshark = MiniTransformerSteinshark(context1_size,context2_size,core_size,lg_size,md_size,n_embed,n_layers,n_heads,n_ff,vocab_size,act_fn,dropout)
    elif args.model_type == "base":
        model:LMSteinshark              = LMSteinshark(core_size,n_embed,n_layers,n_heads,n_ff,vocab_size,act_fn,dropout)
    
    model.name                  = args.model_name
    model                       = model.bfloat16()
    if eval(args.load):
        model.load(root=MODELS)

    #model                       = model.bfloat16()
    print(f"generated model\n\n{model.model_info()}\n\n")
    MODEL                       = model
    TOKENIZER                   = tokenizer


    #Create optimizer 
    optimizer                   = torch.optim.AdamW(params=model.parameters(),lr=lr,weight_decay=wd,betas=(.95,.99))
    lr_sched                    = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr,pct_start=pct_start,total_steps=lr_steps,div_factor=10,final_div_factor=4)

    #Create updates 
    losses                      = [] 
    LOSS                        = losses
    tok_thru_per_iter           = []
    time_per_iter               = [] 

    #Train model 
    cur_train_iter              = MODEL.stats['iter_through']
    train_iters                 += cur_train_iter
    trainset_iter               = 0
    trainset_tok                = 0

    print(f"Beginning training\n\tModel Size:\t{model.n_params//1_000_000}M params\n\tData Size:\t{dataset.n_tokens//1_000_000}M Tokens\n\tBatch Size:\t{bs}")
    plt.ion()
    plt.show()
    plt.rcParams["figure.raise_window"]=False
    model.stats['time_start'] = time.time()

    # #Speed 
    # scaler                            = torch.amp.grad_scaler.GradScaler('cuda')
    start_time                          = time.time()
    while cur_train_iter < train_iters:

        root.update_idletasks()
        root.update()
        #cur_train_iter                = cur_train_iter + model.stats['iter_through']
        t0                              = time.time()

        #Load data
        num_tok                         = input_size#input_size #+int(int(random.random() < .5)*(1024-input_size)*random.random())
        batch                           = dataset.sample(bs,input_size,model.device,True)

        input_ids                       = batch['input_ids']
        target_ids                      = batch['target_ids']

        #Put through model 
        #with torch.amp.autocast('cuda'):
        logits,target_ids           = model.forward(input_ids,target_ids)
        logits                      = logits.view(bs*core_size,vocab_size)
        targets                     = target_ids.view(bs*core_size)
        #Compute and backward loss 
        loss:torch.Tensor           = torch.nn.functional.cross_entropy(logits, targets) / accu_steps


        ##loss                        = scaler.scale(loss)
        #loss.backward() 
        model.stats['iter_through'] += 1
        model.stats['tok_through']  += float(bs*core_size)
        trainset_iter               += 1
        trainset_tok                += float(bs*core_size)


        if cur_train_iter % 10 == 0:
            #for loss, use a value of 
            model.set_generate_mode()
            with torch.no_grad():
                test_inputs                 = input_ids[:2,:]
                test_targets                = target_ids[:2,:]
                logits,targets              = model(test_inputs,test_targets)
                logits                      = logits.view(test_inputs.size(0)*core_size,vocab_size)
                targets                     = targets.view(test_targets.size(0)*core_size)
                test_loss                   = torch.nn.functional.cross_entropy(logits, targets)
            model.set_train_mode()
            model.stats['tok_snap'].append(model.stats['tok_through'])
            model.stats['losses'].append(float(test_loss.detach()))
        
        #Update for all
        CUR_STEP                = model.stats['iter_through']
        TOT_STEP                = train_iters
        TOK_THRU                = trainset_tok / (time.time() - start_time)
        LOSS                    = model.stats['losses']

        tok_thru_per_iter.append(targets.numel())
        time_per_iter.append(time.time()-t0)

        #Zero if on step cycle 
        if cur_train_iter % accu_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),MAX_NORM)
            optimizer.step()
            optimizer.zero_grad()
            try:
                lr_sched.step()
            except ValueError:
                pass #happens if were at the end
            #Save to stats root
            stats_root              = f"{train_root}/prev/"
            save_dir    = os.path.join(stats_root,f"{model.name}.json")
            tok         = model.stats['tok_snap']
            losses      = model.stats['losses']
            with open(save_dir,mode='w') as writefile:
                writefile.write(json.dumps({"tokens":tok,'losses':losses,'name':model.name}))
            
            plt.cla()
            plt.clf()
            colors      = ["yellow","gold","mediumspringgreen","turquoise","deepskyblue","mediumblue","darkviolet","fuchsia","crimson",'gray','black']
            #Plot all stats in save dir 
            for file in os.listdir(stats_root):
                filepath    = os.path.join(stats_root,file)
                try:
                    stats_dict  = json.loads(open(filepath,'r').read())
                except json.decoder.JSONDecodeError as jde:
                    time.sleep(1)
                    stats_dict  = json.loads(open(filepath,'r').read())
                tok         = [tok // 1_000_000 for tok in 
                               stats_dict['tokens']]
                losses      = stats_dict['losses']   

                #Downsample array by 1,000 times 
                newlen      = int(math.sqrt(len(tok)))
                tok         = reduce_arr(tok,newlen)
                losses      = reduce_arr(losses,newlen)

                plt.plot(tok,losses,label=stats_dict['name'],color=colors.pop(0))

            plt.title(f"Model Loss - {model.n_params//1_000_000}M params - [{cur_train_iter}/{train_iters}]")
            plt.xlabel("Tok. Trained On (Millions)")
            plt.ylabel("Batch Loss")
            plt.legend()
            plt.draw()
            plt.pause(.001)

        #Check if new epoch 
        if model.stats['tok_through'] // dataset.n_tokens > model.stats["eps_through"]:
            model.stats["eps_through"] += 1

        print_update(cur_train_iter,train_iters,model.stats['losses'],TOK_THRU,model,sample_text,tokenizer,optimizer,args)

        cur_train_iter += 1 



