from tokenizers.implementations import ByteLevelBPETokenizer
import torch
from torch.utils.data import Dataset 
import numpy
import os 
import random 


#Loads a tokenizer from f_root
def load_tokenizer(f_root:str)->ByteLevelBPETokenizer:
    tokenizer               = ByteLevelBPETokenizer().from_file(vocab_filename=f"{f_root}/vocab.json",merges_filename=f"{f_root}/merges.txt")
    print(f"init tokenizer size {tokenizer.get_vocab_size()}")
    tokenizer.add_tokens(["<|endoftext|>"])
    print(f"loaded tokenizer size {tokenizer.get_vocab_size()}")
    return tokenizer


#Allows sampling of tokens
class TokenizedDataset(Dataset):

    def __init__(self, tokens, n_positions,max_tokens=None,shuffle=False):
        
        #Treat it as tokens list 
        if isinstance(tokens, numpy.ndarray):
            tokens = torch.from_numpy(tokens)
        
        #Treat it as folder root path 
        elif isinstance(tokens,str):
            tokens          = numpy.asarray([])
            self.n_tokens   = 0 

            #Get files loaded
            files           = [os.path.join(tokens,name) for name in os.listdir(tokens) if name.endswith('.npy')]
            if shuffle:
                random.shuffle(files)

            #Grab tokens
            for fname in files:
                tokens += numpy.load(fname)
                self.n_tokens = len(tokens)

                if max_tokens and self.n_tokens > max_tokens:
                    break 
        

        self.tokens         = tokens.contiguous().to(torch.int16)  # Make sure it's contiguous for fast slicing
        self.n_positions    = n_positions
        self.n_tokens       = len(self.tokens)


    #Create indices for sampling
    def build_idxs(self,bs,n_tokens):
        end_point = len(self.tokens) - (n_tokens + 1)
        return  torch.randint(0, end_point, (bs,))

    #Crunch the indices for slicing the final tokens list
    def stack_indices(self,n_tokens,idxs):
        offsets         = idxs.unsqueeze(1) + torch.arange(n_tokens).unsqueeze(0)
        batch_input     = self.tokens[offsets]
        batch_target    = self.tokens[offsets + 1]

        return batch_input,batch_target
    
    #Samples to return tokens of shape (bs,n_tokens) and 
    # place them on device, 
    def sample(self, bs: int, n_tokens: int, device=None) -> dict[str, torch.Tensor]:

        idxs                                = self.build_idxs(bs,n_tokens)

        
        batch_input,batch_target           = self.stack_indices(n_tokens,idxs)
        

        return {
            "input_ids": batch_input.to(device).long(),
            "target_ids": batch_target.to(device).long(),
        }

    def __len__(self):
        return self.n_tokens // self.n_positions



if __name__ == "__main__":

    assert os.path.exists("tokens"), "'tokens' dir does not exist - cannot load tokens"