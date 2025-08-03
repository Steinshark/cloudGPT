import torch 
from collections import OrderedDict
import math
import os 
import json
from tokenizers.implementations import ByteLevelBPETokenizer
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
import time 

os.environ['TORCH_USE_CUDA_DSA'] = "True"


class swiglu(torch.nn.Module):
    
    def __init__(self,d_in,d_out):
        super().__init__()

        self.w1         = torch.nn.Linear(d_in,d_out)
        self.w2         = torch.nn.Linear(d_in,d_out)

    
    def forward(self,x:torch.Tensor)->torch.Tensor:

        return torch.nn.functional.silu(self.w1(x)) * self.w2(x)

#Used to apply RoPE to MHA
def apply_rope(x, seq_len, device):

    head_dim = x.size(-1)
    half_dim = head_dim // 2

    # Compute frequencies
    theta = 1.0 / (10000 ** (torch.arange(0, half_dim, dtype=torch.bfloat16, device=device) / half_dim))
    seq_idx = torch.arange(seq_len, device=device)#.float()
    freqs = torch.einsum("i,j->ij", seq_idx, theta)

    sin = freqs.sin()[None, None, :, :]
    cos = freqs.cos()[None, None, :, :]

    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return x_rotated


#Implementation of multi-head attention
class MultiHeadAttention(torch.nn.Module):
    
    
    def __init__(self, embed_dim, num_heads,n_positions, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),droput=.2):

        super(MultiHeadAttention, self).__init__()
        
        # Initialize parameters
        self.n_positions        = n_positions
        self.embed_dim          = embed_dim
        self.num_heads          = num_heads
        self.d_k                = embed_dim // num_heads
        self.device             = device

        # Linear layers for transforming inputs
        self.layer_1            = torch.nn.Linear(embed_dim,embed_dim*3,bias=True)
        self.W_o                = torch.nn.Linear(embed_dim, embed_dim,device=device,bias=True)
        self.scale              = 1 / math.sqrt(self.d_k)
        self.dropout            = droput
        self.is_training        = True
        

    def forward(self, x:torch.Tensor):
        B, N, C         = x.size()

        # Apply linear transformations and split heads
        Q,K,V           = self.layer_1(x).split(self.embed_dim,dim=2)
        Q:torch.Tensor  = Q.view(B, N, self.num_heads, self.d_k).transpose(1,2)
        K:torch.Tensor  = K.view(B, N, self.num_heads, self.d_k).transpose(1,2)
        V:torch.Tensor  = V.view(B, N, self.num_heads, self.d_k).transpose(1,2)

        # Apply RoPE
        Q               = apply_rope(Q,N,self.device)
        K               = apply_rope(K,N,self.device)

        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,SDPBackend.FLASH_ATTENTION,SDPBackend.MATH,SDPBackend.CUDNN_ATTENTION]):
            attn_out    = scaled_dot_product_attention(Q,K,V,dropout_p=self.dropout if self.is_training else 0,is_causal=True,scale=self.scale)
            attn_out    = attn_out.transpose(1,2).contiguous().view(B,N,C)
            return self.W_o(attn_out)
      

#One stack of a decode-transformer layer
class DecoderLayer(torch.nn.Module):

    def __init__(self,n_embed,n_head,n_positions,n_ff,dropout=.1):
        super(DecoderLayer,self).__init__()
        
        #Self attention layer
        self.mh_attn                = MultiHeadAttention(n_embed,n_head,n_positions)
        self.mha_dropout            = torch.nn.Dropout(p=dropout)
        self.mha_layer_norm         = torch.nn.LayerNorm(n_embed)
        
        #Feed Forward layer
        self.ff_layers              = torch.nn.Sequential(torch.nn.Linear(n_embed,n_ff), torch.nn.GELU(), torch.nn.Linear(n_ff,n_embed)) #swiglu(n_ff,n_ff)
        self.ff_dropout             = torch.nn.Dropout(p=dropout)
        self.ff_layer_norm          = torch.nn.LayerNorm(n_embed)
        
   
    def forward(self,x:torch.Tensor)->torch.Tensor:
        
        #Apply layer_norm, MHA, and residual connection
        attn_output                 = self.mh_attn(self.mha_layer_norm(x))
        attn_output                 = self.mha_dropout(attn_output)
        x                           = x + attn_output

        #Apply layer_norm, ff_layer, and residual connection
        ff_output                   = self.ff_layers(self.ff_layer_norm(x))
        ff_output                   = self.ff_dropout(ff_output)
        x                           = x + ff_output

        return x

 

class LMSteinshark(torch.nn.Module):


    def __init__(self,
                 n_positions:int=512,
                 n_embed    :int=512,
                 n_layers   :int=16,
                 n_heads    :int=16,
                 n_ff       :int=1024,
                 n_vocab    :int=32768,
                 act_fn     :torch.nn.functional=torch.nn.GELU,
                 dropout    :float=.1):
        
        #Superclass it
        super(LMSteinshark,self).__init__()
        
        #Ensure embedding dim is equally divisible into n_heads
        assert n_embed % n_heads == 0

        #Set class variables
        self.n_positions            = int(n_positions)
        self.n_embed                = int(n_embed)
        self.n_layers               = int(n_layers)
        self.n_heads                = int(n_heads)
        self.n_ff                   = int(n_ff)
        self.device                 = torch.device('cuda:0')
        try:
            dropout                 = float(dropout)
        except ValueError:
            dropout                 = .1

        #Use only vocab embeddings
        self.embeddings             = torch.nn.Embedding(n_vocab,n_embed).to(self.device)

        #Create decoder stacks 
        self.transformer_stack      = torch.nn.Sequential(
            OrderedDict(
                {str(i):DecoderLayer(n_embed,n_heads,n_positions,n_ff,dropout=dropout) 
                 for i in range(n_layers)})).to(self.device)

        #Layer norm one last time on the LM Head. Weights are tied to embeddings
        self.output_ln              = torch.nn.LayerNorm(n_embed).to(self.device)

        #Calc params 
        self.n_params               = sum(p.numel() for p in self.parameters())

        #Save name 
        self.name                   = "LMStein"

        #Stats 
        self.stats                  = {"iter_through":0,
                                       "tok_through":0,
                                       "losses":[],
                                       "tok_snap":0,
                                       "time_snap":time.time()}
        #Init weights 
        self.initialize_weights()
        
    
    def forward(self,input_ids:torch.Tensor,target_ids:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:

        #Embed tokens        
        x                               = self.embeddings(input_ids)

        #Pass through transformer stack
        x                               = self.transformer_stack(x)

        #Pass thorugh lm head
        x                               = self.output_ln(x)
        logits                          = x @ self.embeddings.weight.T

        return logits, target_ids

   
    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.kaiming_normal_(param)


    #Creates a json file with the following parameters:
    def save(self, root="C:\\data\\nlp\\models",save_weights=False):
        save_path = os.path.join(root, self.name)

        # Create the directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save the model weights
        if save_weights:
            torch.save(self.state_dict(), os.path.join(save_path, "model_weights.pth"))

        # Prepare metadata to save
        metadata = {
            "n_positions": self.n_positions,
            "n_embed": self.n_embed,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_ff": self.n_ff,
            "n_vocab": self.embeddings.num_embeddings,
            "act_fn": self.transformer_stack[0].act_fn.__class__.__name__ if hasattr(self.transformer_stack[0], 'act_fn') else str(self.transformer_stack[0]),
            "dropout": self.transformer_stack[0].dropout if hasattr(self.transformer_stack[0], 'dropout') else "unknown",
            "stats": self.stats
        }

        # Save the metadata as a JSON file
        with open(os.path.join(save_path, "model_config.json"), "w") as f:
            json.dump(metadata, f, indent=4)


    def load(self, root="C:\\data\\nlp\\models",freeze_loaded=True):
        load_path = os.path.join(root, self.name)

        # Load metadata/config
        config_path = os.path.join(load_path, "model_config.json")
        with open(config_path, "r") as f:
            metadata = json.load(f)

        # Restore stats
        self.stats = metadata.get("stats", {})

        # Check architecture compatibility
        # assert metadata["n_embed"] == self.n_embed, "Mismatch in n_embed"
        # assert metadata["n_layers"] == self.n_layers, "Mismatch in n_layers"
        # assert metadata["n_heads"] == self.n_heads, "Mismatch in n_heads"
        # assert metadata["n_ff"] == self.n_ff, "Mismatch in n_ff"
        # assert metadata["n_vocab"] == self.embeddings.num_embeddings, "Mismatch in vocab size"

        # Load weights
        weights_path = os.path.join(load_path, "model_weights.pth")

        #Generate compatible state dict 
        stored_weights          = torch.load(weights_path, map_location=self.device)
        model_weights           = self.state_dict()
        compatible_weights      = {k:v for k,v in stored_weights.items() if k in self.state_dict() and v.size() == model_weights[k].size()}
        try:
            self.load_state_dict(compatible_weights,strict=False)
            print(f"loaded {len(compatible_weights)}/{len(model_weights)} modules")

            if freeze_loaded:
                for key,module in compatible_weights.items():
                    model_weights[key]._requires_grad = False

        except RuntimeError as re:
            print(f"unable to load weights")

        #print(f"[INFO] Model loaded successfully from: {load_path}")


    @staticmethod
    def from_loadpoint(load_path:str):
        """
        Static method to instantiate LMSteinshark from a saved config and weights.

        Args:
            load_path (str): Path to the saved model directory containing
                             'model_config.json' and 'model_weights.pth'.

        Returns:
            LMSteinshark: A fully initialized and weight-loaded instance.
        """
        # Load metadata/config
        config_path = os.path.join(load_path, "model_config.json")
        with open(config_path, "r") as f:
            metadata = json.load(f)

        # Map string activation function to actual torch class
        act_fn_str = metadata.get("act_fn", "GELU").lower()
        act_fn_map = {
            "gelu": torch.nn.GELU,
            "relu": torch.nn.ReLU,
            "silu": torch.nn.SiLU,
            "tanh": torch.nn.Tanh,
            "leakyrelu": torch.nn.LeakyReLU
        }
        act_fn = act_fn_map.get(act_fn_str.lower(), torch.nn.GELU)

        # Instantiate the model
        model = LMSteinshark(
            n_positions = metadata["n_positions"],
            n_embed     = metadata["n_embed"],
            n_layers    = metadata["n_layers"],
            n_heads     = metadata["n_heads"],
            n_ff        = metadata["n_ff"],
            n_vocab     = metadata["n_vocab"],
            act_fn      = act_fn,
            dropout     = metadata.get("dropout", 0.1)
        )

        # Load stats
        model.stats = metadata.get("stats", {})

        # Load model weights
        weights_path = os.path.join(load_path, "model_weights.pth")
        model.load_state_dict(torch.load(weights_path, map_location=model.device))

        print(f"[INFO] Loaded LMSteinshark from {load_path} with {model.n_params:,} parameters.\n{model}")

        return model


    def set_generate_mode(self):
        self.eval()
        for decoder_layer in self.transformer_stack:
            decoder_layer.mh_attn.is_training = False 


    def set_train_mode(self):
        for decoder_layer in self.transformer_stack:
            decoder_layer.mh_attn.is_training = True
        self.train()


    def generate(self,prompt:list[int],tokenizer:ByteLevelBPETokenizer,n_tokens=128,temperature=.5,top_k=30):
        
        self.set_generate_mode()
        
        with torch.no_grad():
            tokens          = prompt
            model_output    = []
        
            while len(tokens) - len(prompt) < n_tokens:

                input_seq                       = torch.tensor(tokens[-self.n_positions:],device=self.device).long().unsqueeze_(0)
                target_ids                      = torch.empty_like(input_seq)
                logits                          = self(input_seq,target_ids)[0][0,-1,:].float()

                    #Scale logits by temp 
                logits                          = logits / temperature
                vals,indices                    = torch.topk(logits,k=top_k)

                distribution                    = torch.nn.functional.softmax(vals,dim=-1)
                local_i                         = torch.distributions.Categorical(probs=distribution).sample()
                
                next_token                      = indices[local_i]

                #Stop with end seq
                if next_token == tokenizer.encode("<|endoftext|>").ids[0]:
                    self.train()
                    return model_output
                
                model_output.append(next_token)
                tokens                          = tokens + [next_token]

        
        self.set_train_mode()
        return model_output


    def token_streamer(self,prompt:list[int],tokenizer:ByteLevelBPETokenizer,n_tokens=128,temperature=.7,top_k=100):

        self.set_generate_mode()
        full_token_list         = prompt
        generated_token_list    = []

        while len(generated_token_list) < n_tokens:

            #Get model output
            model_input         = torch.tensor(full_token_list).cuda().long().unsqueeze(dim=0)  
            placeholder_ids     = torch.zeros_like(model_input).cuda().long()
            logits,_            = self(model_input,placeholder_ids)

            logits              = logits[0,-1,:].float()
            logits                          = logits / temperature
            vals,indices                    = torch.topk(logits,k=top_k)

            distribution                    = torch.nn.functional.softmax(vals,dim=-1)
            local_i                         = torch.distributions.Categorical(probs=distribution).sample()
            next_token                      = indices[local_i]

            #Check if end of sequence
            if next_token == tokenizer.encode('<|endoftext|>').ids[0]:
                break
            else:
                full_token_list.append(next_token)
                generated_token_list.append(next_token)
                yield tokenizer.decode([next_token])

        self.set_train_mode()
        return 


    def model_info(self) -> str:
        info    = f"Model:\t{self.name}\n"

        info    += f'\tparameters:\t{self.n_params // 1_000_000}M\n'
        info    += f'\tnum layers:\t{self.n_layers}\n'
        info    += f'\tcontext:\t{self.n_positions}\n'
        info    += f'\tembed dim:\t{self.n_embed}\n'
        info    += f"\tff size:\t{self.n_ff}\n"
        info    += f'\tnum heads:\t{self.n_heads}\n'
        info    += f'\ttrain dtype:\t{str(self.embeddings.weight.dtype).replace("torch.","")}'
        

        return info



if __name__ == "__main__":
    #Ensure optimizations 
    torch.backends.cuda.matmul.allow_tf32   = True
    torch.backends.cudnn.allow_tf32         = True
    n_embed     = 2048
    n_ff        = n_embed*4
    n_heads     = n_embed//256
    bs          = 2
    n_positions = 1024
    n_vocab     = 32768
    n_layers    = 12

    #Create model
    lm          = LMSteinshark(n_layers=n_layers,n_embed=n_embed,n_heads=n_heads,n_positions=n_positions,n_vocab=n_vocab,n_ff=n_ff,dropout=0)
    lm.name     = '12x2048[256]x1024x4c'
    scaler                      = torch.amp.GradScaler(enabled=True)
    import training
    print(f"loading")
    lm.load(training.MODELS)
    lm.float()
    print(f"done")
    print(lm.model_info())
    import tok 
    t           = tok.load_tokenizer(f'{training.PATH}/32k_c++') 
    while True:
        lm(torch.zeros(size=(2,n_positions),dtype=torch.long).cuda(),torch.zeros(size=(2,n_positions),dtype=torch.long).cuda())
        prompt      = input('user: ')
        tokens      = t.encode(prompt).ids
        output      = t.decode(lm.generate(tokens,t,64,.7,top_k=100,scaler=scaler))
        print(f"\n\nmodel:{output}\n\n\n")
    exit()
