import os 
import json
import string 
import time
PATH                    = ""

#Establish tracker file for already downloaded wet files 
TOK_PATH                = f"tokens"

PREV_RUNS               = f"prev"

MODELS                  = f"models"

#ENV_PREFIX              = f"/home/ubuntu/FactTune/nlp"
ENV_PREFIX              = f"//Steinpc/s/nlp"

for fpath in [TOK_PATH,PREV_RUNS,MODELS]:
    if not os.path.exists(fpath):
        os.mkdir(fpath)

END_TOKEN                       = "<|endoftext|>"

ALLOWABLE_CHAR                  = string.ascii_lowercase + string.ascii_uppercase + "1234567890!@#$%^&*()~`':;{[}]_-+=<,>.?/}|\\ \n\t" + '"'

LOWER                           = False


#TRAINING SETTINGS 
UPDATE_EVERY_T                  = int(60)
SAMPLE_EVERY_T                  = 10*60
LAST_UPDATE_T                   = time.time() 
LAST_SAMPLE_T                   = time.time() - int(.3*SAMPLE_EVERY_T)
MAX_NORM                        = 2
TOKENIZER                       = 'tokenizer/'
PROMPT                          = "<|endoftext|>"
UPDATE_FREQ                     = 50
SAVE_FREQ                       = 10_000
LAST_SAVE                       = 0 
PROMPT                          = "Computer scientists have developed a novel way to train neural networks."
