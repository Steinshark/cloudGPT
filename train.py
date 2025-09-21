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

def build_validation_set(tokens:torch.Tensor,context_size:int):
    
    tokens          = tokens.cpu().numpy()
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
def print_update(cur_train_iter,train_iters,model:LMSteinshark,tokenizer:ByteLevelBPETokenizer,optimizer:torch.optim.Adam,args):

    global LAST_UPDATE_T
    global LAST_SAMPLE_T


    #Check to sample 
    if time.time() - LAST_SAMPLE_T > SAMPLE_EVERY_T:
        print(f"\n\nPrompt: {PROMPT}\nModel:",end='')
        model_output            = ''.join(model.token_streamer(tokenizer.encode(PROMPT).ids,TOKENIZER,n_tokens=64,temperature=.7,topk=100,topp=.9,mode='p',verbose=False,tokenized=True))
        print(f"{model_output}\n\n")
        LAST_SAMPLE_T          = time.time()



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
    argparser.add_argument("--n_layers",default='16')
    argparser.add_argument('--bs_tok',default='256*1024')
    argparser.add_argument('--ds_name',default='/home/ubuntu/pretrain/nlp/data/smallset')
    #argparser.add_argument('--ds_name',default='//Steinpc/s/nlp/toyset')
    argparser.add_argument('--tokenizer_name',default='tokenizer')
    argparser.add_argument('--input_size',default='512')
    argparser.add_argument('--model_name',default='preTrain_Small_full')
    argparser.add_argument('--n_embed',default='1024')
    argparser.add_argument('--head_dim',default='64')
    argparser.add_argument('--n_ff',default='4')
    argparser.add_argument('--load',default='False')
    argparser.add_argument('--max_tok',default='50_000_000_000')
    argparser.add_argument("--total_tok",default='4_000_000_000')
    args                        = argparser.parse_args()


    #Load data 
    max_tokens                  = eval(args.max_tok)
    dataset                     = TokenizedDataset(args.ds_name,eval(args.input_size),max_tokens=max_tokens,shuffle=True)
    #Split to a validation set 
    validation_set              = dataset.validation_set
    test_inputs,test_targets    = build_validation_set(validation_set,eval(args.input_size))
    test_inputs                 = test_inputs.cuda()
    test_targets                = test_targets.cuda()
    mask                        = torch.ones_like(test_inputs,device=torch.device('cuda')).bool()

    tokenizer_name              = args.tokenizer_name                           #Tokenizer used
    train_root                  = PATH                                          #Where all the training data will be found  
    tokenizer                   = load_tokenizer(f"{tokenizer_name}")


    #Training/Model Settings 
    input_size                  = eval(args.input_size)                         #1:1 sequence 
    vocab_size                  = tokenizer.get_vocab_size()                    #Vocab Size
    n_embed                     = eval(args.n_embed) 
    total_tok                   = eval(args.total_tok)
    print(f"loaded vocab size {vocab_size}")

    #Model settings 
    n_layers                    = eval(args.n_layers)                           #Transformers stacked 
    n_heads                     = n_embed//eval(args.head_dim)                  #Number of attn heads          
    n_ff                        = int(n_embed*eval(args.n_ff))                  #Size of the feed forward network 
    act_fn                      = torch.nn.GELU                                 #Used throughout model

    #Training settings
    train_batch_tok             = eval(args.bs_tok)                             #Number of tokens before stepping optimizer 
    bs                          = eval(args.bs)                                 #BS used per train iter (NOT per optimizer update)
    assert train_batch_tok % input_size == 0, "Batch Token size must be divisible by input size"
    lr                          = .0002                                         #Max LR used in OneCycleLR
    wd                          = .001                                          #WD used throughout
    dropout                     = .05                                           #P used throughout
    virtual_bs                  = train_batch_tok // input_size                 #Number of iters before stepping Optimizer
    accu_steps                  = virtual_bs // bs                              #Number of steps before stepping optimizer
    pct_start                   = .05                                           #Where peak LR will occur       
    train_iters                 = total_tok // (bs*input_size)                  #Total iters used to train
    lr_steps                    = train_iters // accu_steps                     #Total steps (used for OneCycleLR)
    tokenizer_name              = args.tokenizer_name                           #Tokenizer used
    #Sampling 
    sample_text                 = "Scientists have discovered a new technique for creating Large Language Models."
    PROMPT                      = sample_text
    #Create Tokenizer
    assert tokenizer.get_vocab_size() == vocab_size

    #Create model 
    model:LMSteinshark          = LMSteinshark(input_size,n_embed,n_layers,n_heads,n_ff,vocab_size,act_fn,dropout,dtype=torch.bfloat16)
    
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
    optimizer                   = torch.optim.AdamW(params=model.parameters(),lr=lr,weight_decay=wd,betas=(.9,.999))
    #lr_sched                    = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr,pct_start=pct_start,total_steps=lr_steps,div_factor=10,final_div_factor=10)
    warmup_steps                = lr_steps // 20 #5%
    decay_steps                 = lr_steps - warmup_steps
    warmup_sched                = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=.1,end_factor=1,total_iters=warmup_steps)
    decay_sched                 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=decay_steps,eta_min=lr/100)
    scheduler                   = torch.optim.lr_scheduler.ChainedScheduler([warmup_sched,decay_sched])
    static_mask                 = torch.ones(size=(bs,input_size),dtype=torch.bool,device=model.device)

    #Train model 
    cur_train_iter              = MODEL.stats['iter_through']
    train_iters                 += cur_train_iter
    trainset_iter               = 0
    trainset_tok                = 0

    model_size                  = f"{model.n_params//1_000_000}M params"
    data_size                   = f"{dataset.n_tokens//1_000_000}M Tokens"
    training_size               = f"{(total_tok/1_000_000_000):.2f}B Tokens"
    print(f"Beginning training\n\tModel Size:\t{model_size}\n\tData Size:\t{data_size}\n\tTrain Size:\t{training_size}\n\tBatch Size:\t{bs}")
    
    #Create stat "time_start" to track local training run 
    model.stats['run_time_start']   = time.time()
    model.stats['run_tok_through']  = 0
    model.stats['run_iter_through'] = 0

    model.stats['tok_snap']         = 0             #For measuring thorughput 
    model.stats['time_snap']        = time.time()


    #Implement context increase 
    next_iter                       = [(8*1024,16),(32*1024,64),(64*1024,256),(128*1024,512)]
    cur_context                     = 8
    static_mask                     = torch.ones(size=(bs*input_size//cur_context,cur_context),dtype=torch.bool,device=model.device)
    #Run training loop
    while cur_train_iter < train_iters:
        
        #Check for context size shift 

        if next_iter and next_iter[0][0] < cur_train_iter: 
            old_ctxt                        = cur_context
            cur_context                     = next_iter[0][1]
            next_iter                       = next_iter[1:]
            static_mask                     = torch.ones(size=(bs*input_size//cur_context,cur_context),dtype=torch.bool,device=model.device)
            print(f"\nshift context {old_ctxt} -> {cur_context}\n")
        #Sample data 
        num_tok                             = input_size
        batch                               = dataset.sample(bs,input_size,model.device)

        #Make inputs, targets
        input_ids                           = batch['input_ids']
        target_ids                          = batch['target_ids'] 
        
        #Reshape based on current input size 
        eff_bs                              = bs*input_size//cur_context
        input_ids                           = input_ids.view(eff_bs,cur_context)
        target_ids                          = target_ids.view(eff_bs,cur_context)

        #Put through model 
        logits,target_ids                   = model.forward(input_ids,target_ids,static_mask)
        logits                              = logits.view(bs*input_size,vocab_size)
        targets                             = target_ids.view(bs*input_size)

        #Compute loss and backprop
        loss:torch.Tensor                   = torch.nn.functional.cross_entropy(logits, targets) / accu_steps
        loss.backward()

        #Update for rate tracking 
        model.stats['tok_snap']             += input_ids.numel()

        #Update stats
        model.stats['iter_through']         += 1
        model.stats['run_iter_through']     += 1

        model.stats['tok_through']          += input_ids.numel()
        model.stats['run_tok_through']      += input_ids.numel()

        #Zero if on step cycle 
        if (cur_train_iter % accu_steps) == 0 and not cur_train_iter == 0:

            #Clip norm and step
            torch.nn.utils.clip_grad_norm_(model.parameters(),MAX_NORM)
            optimizer.step()
            optimizer.zero_grad()
                        
            #Step lr_scheduler (with error at very end)
            scheduler.step()

            #Save to stats root
            saving_weights  = (cur_train_iter - LAST_SAVE) > SAVE_FREQ
            model.save(root=f"{MODELS}",save_weights=saving_weights)
            if saving_weights:
                print(f"\tsaved weights")
                LAST_SAVE = cur_train_iter           

            #Get validation loss
            with torch.inference_mode():
                model.set_generate_mode()
                logits,targets              = model(test_inputs,test_targets,mask)
                logits                      = logits.view(test_inputs.size(0)*input_size,vocab_size)
                targets                     = targets.view(test_targets.size(0)*input_size)
                test_loss                   = torch.nn.functional.cross_entropy(logits, targets)
            model.set_train_mode()
            model.stats['losses'].append(float(test_loss))
            iter_str    = f"             [{cur_train_iter}/{train_iters}]"[-17:]
            loss_str    = f'    loss={test_loss:.7f}0000'[:16]

            token_throughput = model.stats['tok_snap']/(time.time()-model.stats['time_snap'])

            tok_thru    = f"       tok/sec={(token_throughput/1_000):.1f}K"[-16:]
            toks        = f"          {model.stats['tok_through']/1_000_000:.1f}M tokens"[-16:]
            lr          = f"        lr={optimizer.param_groups[0]['lr']:.8f}"[-16:]

            wall_t_left = f"        t_remain={((total_tok-model.stats['tok_through']) / (token_throughput*3600)):.1f}h"
            print(f"[PARAM UPDATE]{iter_str}{loss_str}{tok_thru}{toks}{lr}{wall_t_left}")

        
        print_update(cur_train_iter,train_iters,model,tokenizer,optimizer,args)


        cur_train_iter += 1 

    #We're done!
    model.save(root=f"{MODELS}",save_weights=True)
    print(f"Model has finished training")

