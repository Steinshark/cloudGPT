import numpy 

def reduce_arr(arr:list,newlen:int):
    if not arr:
        return []
    
    newlen      = max(1,newlen)
    arr         = numpy.asarray(arr)
    factor      = len(arr) // newlen
    reduced     = arr[-newlen*factor:].reshape(newlen, factor).mean(axis=1)
   
    return reduced

END_TOKEN           = "<|endoftext|>"
CODE_TOKEN          = "<|writecode|>"
RUNCODE_TOKEN       = "<|runcode|>"
WEB_TOKEN           = '<|websearch|>'
PROMPT_TOKEN        = '<|prompt|>'
RESPONSE_TOKEN      = '<|response|>'
RESERVE_1           = '<|reserve1|>'
RESERVE_2           = '<|reserve2|>'


SPECIAL_TOKENS      = [END_TOKEN,
                       CODE_TOKEN,
                       RUNCODE_TOKEN,
                       WEB_TOKEN,
                       PROMPT_TOKEN,
                       RESPONSE_TOKEN,
                       RESERVE_1,
                       RESERVE_2]