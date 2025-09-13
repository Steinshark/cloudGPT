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


SPECIAL_TOKENS      = {"eot":END_TOKEN,
                       "code":CODE_TOKEN,
                       "runcode":RUNCODE_TOKEN,
                       "web":WEB_TOKEN,
                       "prompt":PROMPT_TOKEN,
                       "resp":RESPONSE_TOKEN,
                       "r1":RESERVE_1,
                       "r2":RESERVE_2}

# Generate 100 unique, varied refutation phrases to answer unanswerable SQuAD2.0-style questions
refuatations = [
    "There is no information in the passage to answer this question.",
    "The passage does not contain the answer to this question.",
    "This question cannot be answered based on the passage.",
    "The required information is not present in the provided context.",
    "The context does not provide a relevant answer.",
    "The passage lacks the necessary details to respond to this question.",
    "No supporting evidence is given in the text.",
    "The text does not address this particular question.",
    "This information is not mentioned in the passage.",
    "The passage fails to mention anything about this topic.",
    "There is no mention of this subject in the context.",
    "The passage does not specify anything about this.",
    "The context offers no insight into this matter.",
    "There are no relevant facts provided to answer this.",
    "This detail is absent from the passage.",
    "The context does not include this information.",
    "No details are provided on this topic.",
    "The passage provides no such information.",
    "This is not covered in the provided text.",
    "Nothing in the passage relates to this question.",
    "The answer cannot be derived from the context.",
    "This topic is outside the scope of the passage.",
    "The passage offers no answer to this question.",
    "This question is not supported by the text.",
    "There is no content in the passage that supports an answer.",
    "This issue is not addressed in the passage.",
    "The passage contains no information relevant to this question.",
    "This is not discussed in the context.",
    "The provided context does not include this information.",
    "No content in the text pertains to this.",
    "The answer is not inferable from the passage.",
    "The topic is not explored in the context.",
    "This question goes beyond the information given.",
    "The passage is silent on this matter.",
    "This information is not included in the passage.",
    "The question asks for details not found in the text.",
    "The answer is not mentioned or implied in the passage.",
    "The passage gives no answer to this inquiry.",
    "This is not mentioned anywhere in the context.",
    "This data is not available in the given passage.",
    "The passage doesn't touch on this subject.",
    "The context lacks details about this.",
    "The information required is missing from the text.",
    "No relevant information is presented in the passage.",
    "The passage doesn't contain any clues to answer this.",
    "The context doesn't discuss this issue.",
    "This cannot be determined from the provided context.",
    "The question has no answer in the passage.",
    "The passage has no bearing on this question.",
    "This is outside the information given in the text."
]
