import os 
import json
from tokenizers.implementations import ByteLevelBPETokenizer
import crawl 
from training import *
import numpy
import time 
import multiprocessing
import random 
from dataset import TokenizedDataset, load_tokenizer


#Tokenizes based on the tokens found in CRAWL_DB
def train_tokenizer(vocab_size:int,name:str,db:str=CRAWL_DB) ->ByteLevelBPETokenizer:
    print(f"Training {name} tokenizer size={vocab_size}")
    tokenizer               = ByteLevelBPETokenizer()
    tokenizer.train([os.path.join(db,fname) for fname in os.listdir(db)],vocab_size=vocab_size-1)
    tokenizer.add_tokens([END_TOKEN])

    if not os.path.exists(f"{PATH}/{name}"):
        os.mkdir(f"{PATH}/{name}")

    tokenizer.save_model(f"{PATH}/{name}")
    print(f"\tcomplete - saved as {name}")
    

#Loads tokenizer from default location. Adds the endoftext token
def load_tokenizer(tokenizer_name:str)->ByteLevelBPETokenizer:
    tokenizer               = ByteLevelBPETokenizer().from_file(vocab_filename=f"{tokenizer_name}/vocab.json",merges_filename=f"{tokenizer_name}/merges.txt")
    tokenizer.add_tokens([END_TOKEN])
    return tokenizer



def tokenize_save_text(data:dict):

    #Load contents
    tokenizer   = data['tokenizer']
    fpath       = data['fpath']
    db          = data['db']
    tok_db      = data['tok_db']

    tokenizer   = load_tokenizer(tokenizer)
    with open(fpath,'r',encoding='utf_8') as readfile:
        contents    = readfile.read()
        if LOWER:
            contents= contents.lower()

        #fix contents
        for rep_word in ["the", "and", " is", "are", "of"]:
            contents    = contents.replace(f" {rep_word} {rep_word} ", f" {rep_word} ")

        print(f"tokenizing {fpath}")
        ids         = tokenizer.encode(contents).ids
        np_ids      = numpy.asarray(ids).astype(numpy.uint16)
        tokpath     = fpath.replace(db,tok_db).replace(".txt",".npy")

        numpy.save(tokpath,np_ids)
    return len(np_ids)


#Tokenizes the corpus found in CRAWL_DB and saves it to TOK_DB
def tokenize_corpus(tokenizer_name:str,db:str=CRAWL_DB,tok_db:str=TOK_DB,n_workers:int=4):

    corpus      = [os.path.join(db,fname) for fname in os.listdir(db)]

    args        = [] 

    for fpath in corpus:
        tokpath     = fpath.replace(db,tok_db).replace(".txt",".npy")

        #Skip if weve done it
        if os.path.exists(tokpath):
            continue

        args.append(({'tokenizer':tokenizer_name,'fpath':fpath,'db':db,'tok_db':tok_db}))
    
    with multiprocessing.Pool(processes=n_workers) as pool:

       results      = pool.map(tokenize_save_text,args)
    

    total_tok   = 0 
    for res in results:
        total_tok += res 

    print(f"generated {total_tok/1_000_000_000:.3f}B tokens")


def load_tokens(args,max_tokens):
       #Load data 
    tokens                      = [] 
    n_tok_loaded                = 0
    fnames                      = [fname for fname in os.listdir(f"{args.train_root}/{args.ds_name}")]
    fnames.sort(key= lambda x: int(x.replace("tokens","").replace(".npy","").replace(".txt","")))
    for fname in fnames:
        fname               = f"{args.train_root}/{args.ds_name}/{fname}"
        newtok:numpy.array  = numpy.load(fname).astype(numpy.uint16)
        tokens.append(newtok)
        n_tok_loaded        += len(newtok)

        if n_tok_loaded > max_tokens:
            break

    tokens                      = numpy.concatenate(tokens)[-n_tok_loaded:]
    dataset                     = TokenizedDataset(tokens,eval(args.input_size))
    _N_TOKENS                   = dataset.n_tokens

    return dataset,len(tokens)


if __name__ == "__main__":
    name        = f"{PATH}/32k_c++"

    #train_tokenizer(32768,name,db=FINEDB)
    t = load_tokenizer(name)
    #print(f"Steinshark -> {t.encode('Steinshark').ids} -> {t.decode(t.encode('Steinshark').ids)}")
    #exit()
    #Loop so that it always runs
    #while True:
    tokenize_corpus(name,db=FINEDB,tok_db=TOK_DB_CLEAN,n_workers=8)
        #time.sleep(10)