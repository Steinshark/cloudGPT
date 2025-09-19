import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from model import LMSteinshark
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
from data import TokenizedDataset, load_tokenizer
import time 
from matplotlib import pyplot as plt 
from utils import reduce_arr
from environment import *
import numpy 


def unlock_model(model:LMSteinshark,cur_iter:int):
    for i,stack in enumerate(model.transformer_stack):
        if i < cur_iter:
            for param in stack.parameters():
                param.requires_grad_(True)

def lock_model(model:LMSteinshark):

    for stack in model.transformer_stack:
        for param in stack.parameters():
            param.requires_grad_(False)

def build_validation_set(tokens:numpy.array,context_size:int):
    
    inputs          = []
    targets         = [] 
    while len(tokens) > (context_size+1):
        tok_chunk   = tokens[:context_size+1]
        input_ids   = tok_chunk[:-1]
        target_ids  = tok_chunk[1:]
        inputs.append(input_ids)
        targets.append(target_ids)

        tokens      = tokens[context_size+1:]

    inputs          = torch.tensor(numpy.asarray(inputs)).long()
    targets         = torch.tensor(numpy.asarray(targets)).long()
    return inputs,targets 

#               cur_train_iter, train_iters,model,tokenizer,optimizer,args
def print_update(cur_train_iter,train_iters,model:LMSteinshark,tokenizer:ByteLevelBPETokenizer,optimizer:torch.optim.Adam,args,cur_iter:str=1):

    global LAST_UPDATE_T
    global LAST_SAMPLE_T

    #Check for printint stats 
    if time.time() - LAST_UPDATE_T > UPDATE_EVERY_T and False:

        iters                   = "iter " + f"{cur_train_iter}/{train_iters}".rjust(11) + "   "
        losses                  = f"{float(sum(model.stats['losses'][-64:])) / float(len(model.stats['losses'][-64:])+.01):.5f}".rjust(8) + "   "
        tok_thru                = f"{(model.stats['tok_snap']/(time.time()-model.stats['time_snap']))/1_000:.1f}k tok/s" + "   "
        toks                    = f"{model.stats['tok_through']/1_000_000:.1f}M tokens"
        lr                      = f"  lr={optimizer.param_groups[0]['lr']}"
        LAST_UPDATE_T          = time.time()

        model.stats['tok_snap']         = 0 
        model.stats['time_snap']        = time.time()

        print(iters+losses+tok_thru+toks+lr)


    #Check to sample 
    if time.time() - LAST_SAMPLE_T > SAMPLE_EVERY_T:
        print(f"\n\nPrompt: {PROMPT}\nModel:",end='')
        model_output            = ''.join(model.token_streamer_truncated(tokenizer.encode(PROMPT).ids,TOKENIZER,n_tokens=64,temperature=.7,topk=100,topp=.9,mode='p',verbose=False,tokenized=True,cur_iter=cur_iter))
        print(f"{model_output}\n\n")
        LAST_SAMPLE_T          = time.time()

        if dataset.augment_data():
            print(f"\tDataset augmented  {dataset.n_tokens}")


if __name__ == "__main__":
    
    free    = 0
    total   = 90_000_000_000

    while total-free > 20_000_000_000:
        free, total = torch.cuda.mem_get_info(torch.cuda)
        if total-free < 5_000_000_000:
            break
        print(f"awaiting phase completion - {total-free}GB being used")
        time.sleep(300)
    print(f"commencing training")

    #Ensure optimizations 
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    #Handle arguments
    argparser                   = argparse.ArgumentParser()
    argparser.add_argument('--model_dir',default='')
    argparser.add_argument('--model_type',default='base')
    argparser.add_argument('--bs',default='16')
    argparser.add_argument("--n_layers",default='20')
    argparser.add_argument('--bs_tok',default='1024*1024')
    argparser.add_argument('--ds_name',default='/home/ubuntu/Stein2/data')
    argparser.add_argument('--tokenizer_name',default='tokenizer')
    argparser.add_argument('--input_size',default='1024')
    argparser.add_argument('--model_name',default='production')
    argparser.add_argument('--n_embed',default='2048')
    argparser.add_argument('--head_dim',default='256')
    argparser.add_argument('--n_ff',default='4')
    argparser.add_argument('--load',default='False')
    argparser.add_argument('--max_tok',default='50_000_000_000')

    args                        = argparser.parse_args()


    #Load data 
    max_tokens                  = eval(args.max_tok)
    dataset                     = TokenizedDataset(args.ds_name,eval(args.input_size),max_tokens=max_tokens,shuffle=True)
    #Split to a validation set 
    validation_set              = dataset.tokens[:32768]
    dataset.tokens              = dataset.tokens[32768:]
    dataset.n_tokens            = len(dataset.tokens)

    tokenizer_name              = args.tokenizer_name                           #Tokenizer used
    train_root                  = PATH                                          #Where all the training data will be found  
    tokenizer                   = load_tokenizer(f"{tokenizer_name}")


    #Training/Model Settings 
    input_size                  = eval(args.input_size)                         #1:1 sequence 
    vocab_size                  = tokenizer.get_vocab_size()                    #Vocab Size
    n_embed                     = eval(args.n_embed) 
    print(f"loaded vocab size {vocab_size}")

    #Model settings 
    n_layers                    = eval(args.n_layers)                           #Transformers stacked 
    n_heads                     = n_embed//eval(args.head_dim)                  #Number of attn heads          
    n_ff                        = int(n_embed*eval(args.n_ff))                  #Size of the feed forward network 
    act_fn                      = torch.nn.GELU                                 #Used throughout model

    #Training settings
    train_batch_tok             = eval(args.bs_tok)                             #Number of tokens before stepping optimizer 
    bs                          = eval(args.bs)                                 #BS used per train iter (NOT per optimizer update)
    lr                          = .0002                                         #Max LR used in OneCycleLR
    wd                          = .01                                           #WD used throughout
    dropout                     = .1                                            #P used throughout
    virtual_bs                  = train_batch_tok // input_size                 #Number of iters before stepping Optimizer
    accu_steps                  = virtual_bs // bs                              #Number of steps before stepping optimizer
    pct_start                   = .1                                            #Where peak LR will occur       
    train_iters                 = 3* 512_000_000 // (bs*input_size)             #Total iters used to train
    lr_steps                    = 3* 512_000_000 // accu_steps                  #Total steps (used for OneCycleLR)
    tokenizer_name              = args.tokenizer_name                           #Tokenizer used
    unlocks                     = 8     #Every 8 optimizer steps, unlock a new model layer 
    cur_layer                   = 1000

    #Sampling 
    sample_text                 = "Scientists have discovered a new technique for creating Large Language Models."
    PROMPT                      = sample_text
    #Create Tokenizer
    assert tokenizer.get_vocab_size() == vocab_size

    #Create model 
    model:LMSteinshark              = LMSteinshark(input_size,n_embed,n_layers,n_heads,n_ff,vocab_size,act_fn,dropout)
    
    model.name                  = args.model_name
    model                       = model.bfloat16()
    if eval(args.load):
        model.load(root=MODELS)
        print(f"loaded model")
    #model                       = model.bfloat16()
    print(f"Initialized model\n\n{model.model_info()}\n\tdropout:\t{dropout}\n\n")
    MODEL                       = model
    TOKENIZER                   = tokenizer


    #Create optimizer 
    optimizer                   = torch.optim.AdamW(params=model.parameters(),lr=lr,weight_decay=wd,betas=(.95,.99))
    lr_sched                    = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr,pct_start=pct_start,total_steps=lr_steps,div_factor=10,final_div_factor=100)

    #Train model 
    cur_train_iter              = MODEL.stats['iter_through']
    train_iters                 += cur_train_iter
    trainset_iter               = 0
    trainset_tok                = 0

    print(f"Beginning training\n\tModel Size:\t{model.n_params//1_000_000}M params\n\tData Size:\t{dataset.n_tokens//1_000_000}M Tokens\n\tBatch Size:\t{bs}")
    
    #Create stat "time_start" to track local training run 
    model.stats['run_time_start']   = time.time()
    model.stats['run_tok_through']  = 0
    model.stats['run_iter_through'] = 0

    model.stats['tok_snap']         = 0             #For measuring thorughput 
    model.stats['time_snap']        = time.time()

    #Set all params to require grad 
    for param in model.parameters():
        param.requires_grad_(True)


    unlocked                    = True
    #Remove those not training 
    #Run training loop
    while cur_train_iter < train_iters:

        #Sample data 
        num_tok                             = input_size
        batch                               = dataset.sample(bs,input_size,model.device)

        #Make inputs, targets
        input_ids                           = batch['input_ids']
        target_ids                          = batch['target_ids'] 
        
        #Put through model 
        logits,target_ids                   = model.optimized_forward(input_ids,target_ids,torch.ones_like(input_ids,device=model.device,dtype=torch.bool),cur_layer=cur_layer)
        logits                              = logits.view(bs*input_size,vocab_size)
        targets                             = target_ids.view(bs*input_size)

        #Compute loss and backprop
        loss:torch.Tensor                   = torch.nn.functional.cross_entropy(logits, targets) / accu_steps
        loss.backward()

        #Update for rate tracking 
        model.stats['tok_snap']             += int(bs*input_size)

        #Zero if on step cycle 
        if ((cur_train_iter + 1) % accu_steps) == 0:

            #Clip norm and step
            torch.nn.utils.clip_grad_norm_(model.parameters(),MAX_NORM)
            optimizer.step()
            optimizer.zero_grad()

            
            #Step lr_scheduler (with error at very end)
            try:
                lr_sched.step()
            except ValueError:
                pass

            #Save to stats root
            saving_weights  = (cur_train_iter - LAST_SAVE) > SAVE_FREQ
            model.save(root=f"{MODELS}",save_weights=saving_weights)
            if saving_weights:
                print(f"\tsaved weights")
                LAST_SAVE = cur_train_iter

        #if ((cur_train_iter+1) % UPDATE_FREQ) == 0:

            #Update stats
            model.stats['iter_through']         += UPDATE_FREQ
            model.stats['run_iter_through']     += UPDATE_FREQ

            model.stats['tok_through']          += int(bs*input_size) * UPDATE_FREQ
            model.stats['run_tok_through']      += int(bs*input_size) * UPDATE_FREQ
            

            #Get validation loss
            with torch.inference_mode():
                model.set_generate_mode()
                
                test_inputs,test_targets    = build_validation_set(validation_set,input_size)
                logits,targets              = model.optimized_forward(test_inputs.cuda(),test_targets.cuda(),torch.ones_like(test_inputs,device=model.device,dtype=torch.bool),cur_layer=cur_layer)
                logits                      = logits.view(test_inputs.size(0)*input_size,vocab_size)
                targets                     = targets.view(test_targets.size(0)*input_size)
                test_loss                   = torch.nn.functional.cross_entropy(logits, targets)
            model.set_train_mode()
            model.stats['losses'].append(float(test_loss))
            iter_str    = f"             [{cur_train_iter}/{train_iters}]"[-17:]
            loss_str    = f'    loss={test_loss:.5f}0000'[:16]
            tok_thru    = f"       tok/sec={(model.stats['tok_snap']/(time.time()-model.stats['time_snap']))/1_000:.1f}K"[-16:]
            toks        = f"          {model.stats['tok_through']/1_000_000:.1f}M tokens"[-16:]
            lr          = f"        lr={optimizer.param_groups[0]['lr']:.8f}"[-16:]
            print(f"[PARAM UPDATE]{iter_str}{loss_str}{tok_thru}{toks}{lr}\tcur_layer={cur_layer}")

            #Check to lock for first time 
            if cur_train_iter > 512 and unlocked:
                cur_layer = 1
                lock_model(model)
                unlock_model(model,cur_iter=cur_layer)
                unlocked = False



            if not unlocked and (cur_train_iter + 1) // accu_steps == unlocks:
                cur_layer += 1  
                unlock_model(model,cur_layer)
        
        print_update(cur_train_iter,train_iters,model,tokenizer,optimizer,args,cur_iter=cur_layer)


        cur_train_iter += 1 

    #We're done!
    model.save(root=f"{MODELS}",save_weights=True)
    print(f"Model has finished training")

