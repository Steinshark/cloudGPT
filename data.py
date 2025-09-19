from tokenizers.implementations import ByteLevelBPETokenizer
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
import numpy
import os 
import random 
import sys
sys.path.append("C:gitrepos/cloudGPT")
from utils import SPECIAL_TOKENS, PROMPT_TOKEN, RESPONSE_TOKEN, RESERVE_1, END_TOKEN
import torch
import json
import numpy 

#Loads a tokenizer from f_root
def load_tokenizer(f_root:str)->ByteLevelBPETokenizer:
    tokenizer               = ByteLevelBPETokenizer().from_file(vocab_filename=f"{f_root}/vocab.json",merges_filename=f"{f_root}/merges.txt")
    tokenizer.add_tokens(list(SPECIAL_TOKENS.values()))

    tokenizer.special_tokens    = SPECIAL_TOKENS

    return tokenizer


#Allows sampling of tokens
class TokenizedDataset(Dataset):

    def __init__(self, tokens, input_size,max_tokens=None,shuffle=False,augmenting=True,valid_size=32768):
        
        self.loaded_files   = set()
        self.shuffle        = shuffle
        self.max_tokens     = max_tokens
        self.augmenting     = augmenting

        #Treat it as tokens list 
        if isinstance(tokens, numpy.ndarray):
            tokens = torch.from_numpy(tokens)
        
        #Treat it as folder root path 
        elif isinstance(tokens,str):
            self.root_folder    = tokens 
            token_set       = [] 
            self.n_tokens   = 0 

            #Get files loaded
            files           = [os.path.join(self.root_folder,name) for name in os.listdir(self.root_folder) if name.endswith('.npy')]
            if shuffle:
                random.shuffle(files)

            #Grab tokens
            for fname in files:
                try:
                    newset  = numpy.load(fname)
                    self.n_tokens += len(newset) 
                    token_set.append(newset)

                    #Add to loaded files
                    self.loaded_files.add(fname)
                except ValueError:
                    pass

                if self.max_tokens and (self.n_tokens > self.max_tokens):
                    break 

            tokens  = numpy.concatenate(token_set).flatten()
            tokens  = torch.from_numpy(tokens)

        
        self.tokens         = tokens.contiguous().to(torch.long)[:self.max_tokens+valid_size]  # Make sure it's contiguous for fast slicing
        self.tokens         = self.tokens.cuda()
        
        #Build a test set 
        self.validation_set = self.tokens[:valid_size] 
        self.tokens         = self.tokens[valid_size:].cuda()
        self.tokens.to(torch.int16)

        self.input_size     = input_size
        self.n_tokens       = len(self.tokens)

        self.input_idx      = torch.arange(input_size,device=torch.device('cuda'))
        self.target_idx     = torch.arange(input_size,device=torch.device('cuda')) + 1 

        n_batches           = self.n_tokens // (input_size+1)
        self.base_idxs      = torch.arange(0,n_batches,device='cuda',dtype=torch.int32)
        self.shuffle_indices()

            
    #Reshuffle indices - call for a new epoch
    def shuffle_indices(self):
        perm                = torch.randperm(self.base_idxs.shape[0],device='cuda')
        self.shuffled_idxs  = self.base_idxs[perm]
        self.cur_i          = 0 

    #Grab the next set of shuffled indices
    def build_idxs(self,bs):
        
        if self.cur_i + bs > self.shuffled_idxs.size(0):
            self.shuffle_indices()

        #Grab batch 
        idxs            = self.shuffled_idxs[self.cur_i:self.cur_i+bs]
        self.cur_i      += bs 


        return idxs

    #Crunch the indices for slicing the final tokens list
    def stack_indices(self,n_tokens,idxs):

        batch_input     = self.tokens[idxs[:,None] + self.input_idx]
        batch_target    = self.tokens[idxs[:,None] + self.target_idx]

        return batch_input,batch_target
    
    #Samples to return tokens of shape (bs,n_tokens) and 
    # place them on device, 
    def sample(self, bs: int, n_tokens: int, device=None) -> dict[str, torch.Tensor]:

        idxs                                = self.build_idxs(bs)

        
        batch_input,batch_target           = self.stack_indices(n_tokens,idxs)
        

        return {
            "input_ids": batch_input,
            "target_ids": batch_target,
        }


    #This function loads additional numpy files not available before (due to slowly streaming data over scp connection)
    def augment_data(self):
        
        if not self.augmenting:
            return False 
        
        #Get files loaded
        files               = [os.path.join(self.root_folder,name) for name in os.listdir(self.root_folder) if name.endswith('.npy')]
        addl_tokens         = []

        prev_len            = self.n_tokens
        if self.shuffle:
            random.shuffle(files)

        #Grab tokens
        for fname in files:
            if fname in self.loaded_files:
                continue
            try:
                newset:numpy.array      = numpy.load(fname)
                self.n_tokens           += len(newset) 
                addl_tokens.append(newset)

                #Add to loaded files
                self.loaded_files.add(fname)
            except ValueError:
                pass

            if self.max_tokens and (self.n_tokens > self.max_tokens):
                break 
        
        if addl_tokens:
            tokens              = numpy.concatenate(addl_tokens).flatten()  
            tokens              = torch.from_numpy(tokens).type(torch.int32)

            self.tokens         = torch.cat([self.tokens,tokens])
            self.n_tokens       = len(self.tokens)

        return self.n_tokens > prev_len


class FinetuneTokenizer(ByteLevelBPETokenizer):

    def __init__(self,tokenizer:ByteLevelBPETokenizer,max_len:int=2048,padding_tok=RESERVE_1):

        self.base_tokenizer     = tokenizer
        self.pad_token_id       = self.base_tokenizer.encode(padding_tok).ids[0]
        self.eos_token_id       = self.base_tokenizer.encode(END_TOKEN).ids[0]
        self.max_len            = max_len

    def tokenize(self,text:str):

        #Tokenize and truncate
        tokens                  = self.base_tokenizer.encode(text).ids[:self.max_len]
        return tokens 
    
    def batch_tokenize(self,texts:list[str]):

        batch_tokens            = [self.tokenize(text) for text in texts]
        seq_lens                = [len(seq) for seq in batch_tokens]
        batch_len               = max(seq_lens)
        mask                    = numpy.zeros(shape=(len(batch_tokens),batch_len),dtype=numpy.int16)
        tokens                  = numpy.zeros_like(mask,dtype=numpy.int16)
        tokens[:,:]             = self.pad_token_id                  

        for i,seq_len in enumerate(seq_lens):
            tokens[i,:seq_len]  = batch_tokens[i]   #fill tokens
            mask[i,:seq_len]    = 1                 #fill mask
        
        return tokens,mask



#Finetune dataset accepts any 'json_path' that points to a jsonable_file yielding an iterable of strings
# in the format '<prompt>...<response>...' 
class FinetuneDataset(Dataset):
    def __init__(self, json_path, tokenizer:FinetuneTokenizer, max_length=2048,data_cap=1_000_000,concat=True):

        #Load json data. raw_data will be a list of strings.
        with open(json_path, "r", encoding="utf-8") as readfile:
            if json_path.endswith('l'):
                raw_data    = [json.loads(line) for line in readfile.readlines()[:data_cap]]
            else:
                raw_data = json.loads(readfile.read())

        #Shuffle to choose random selection
        if data_cap < len(raw_data):
            random.shuffle(raw_data)
        raw_data            = raw_data[:data_cap]


        #Create vars 
        self.tokenizer      = tokenizer
        self.pad_token_id   = tokenizer.pad_token_id
        self.max_length     = max_length
        self.data           = [] 

        #Append data items together up to 2049 in length (we'll cut off 1 token when generating 
        # the input and target ids)
        if concat:
            batches             = self.pack_examples(raw_data,self.tokenizer,max_length+1)
        else:
            batches             = [tokenizer.tokenize(datapoint) + [tokenizer.eos_token_id] for datapoint in raw_data]

        #Convert to tensors and generate the data splits
        for item in batches:
            
            tokens      = torch.tensor(item).long()
            input_ids   = tokens[:-1]
            target_ids  = tokens[1:]
            
            attn_mask   = torch.ones_like(input_ids).bool()
            attn_mask[input_ids == self.tokenizer.pad_token_id] = False
            self.data.append({'input_ids':input_ids,"target_ids":target_ids,"attn_mask":attn_mask,"length":len(tokens)})

        # Sort by length for bucketing
        self.data.sort(key=lambda x: x["length"])
    
    @staticmethod
    def pack_examples(dataset, tokenizer, max_len=2048):
        buffer = []
        batches = []
        cur_len = 0

        for ex in dataset:
            ids = tokenizer.tokenize(ex) + [tokenizer.eos_token_id]
            if cur_len + len(ids) > max_len:
                
                #buff it out to padded length
                while len(buffer) < max_len:
                    buffer.append(tokenizer.pad_token_id)

                
                batches.append(buffer)
                buffer = []
                cur_len = 0
            buffer.extend(ids)
            cur_len += len(ids)



        if buffer:  # flush remainder
            #buff it out to padded length
            while len(buffer) < max_len:
                buffer.append(tokenizer.pad_token_id)
            batches.append(buffer)
        return batches
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["input_ids"], self.data[idx]["target_ids"], self.data[idx]["attn_mask"],


class BucketedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, bucket_size=200):
        self.batch_size     = batch_size
        self.dataset        = dataset
        self.bucket_size    = bucket_size

        # Make buckets of indices with similar lengths
        self.buckets = [
            list(range(i, min(i + bucket_size, len(dataset))))
            for i in range(0, len(dataset), bucket_size)
        ]

    def __iter__(self):
        all_batches = []
        for bucket in self.buckets:
            random.shuffle(bucket)
            # Break into batches inside each bucket
            for i in range(0, len(bucket), self.batch_size):
                all_batches.append(bucket[i:i + self.batch_size])
        random.shuffle(all_batches)  # Shuffle batch order across buckets
        return iter(all_batches)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def collate_fn(batch, pad_token_id):
    batch       = sorted(batch, key=lambda x: len(x), reverse=True)
    sequences   = pad_sequence(batch, batch_first=True, padding_value=pad_token_id)
    input_ids   = sequences[:,:-1]
    attn_mask   = (input_ids != pad_token_id).bool()
    target_ids  = sequences[:,1:]
    return {"input_ids": input_ids,"target_ids": target_ids, "attention_mask": attn_mask}


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer   = load_tokenizer('//Steinpc/s/nlp/tokenizer')
    ftt         = FinetuneTokenizer(tokenizer,128,RESERVE_1)
    #out         = ftt.batch_tokenize(["This one is a Test","this Two is a test. It is longer"])


    fname       = '//Steinpc/S/nlp/data/factual_dataset_select.jsonl'
    dataset     = FinetuneDataset(fname, ftt,data_cap=64)

    sampler = BucketedBatchSampler(dataset, batch_size=4, bucket_size=200)

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=lambda x: collate_fn(x, dataset.pad_token_id)
    )

    print(f"loader is {len(loader)}")
    for batch in loader:
        print(batch["input_ids"].shape, batch["attention_mask"].shape)
        print(tokenizer.decode(batch['input_ids'][0].numpy()))
        print(tokenizer.decode(batch['input_ids'][1].numpy()))
        input(tokenizer.decode(batch['input_ids'][2].numpy()))

