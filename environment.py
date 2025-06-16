import os 
import json
import string 
import enchant
import time
PATH                = r""

#Establish tracker file for already downloaded wet files 
TOK_PATH              = f"{PATH}/tokens"

PREV_RUNS           = f"{PATH}/prev"

MODELS              = f"{PATH}/models"



for fpath in [TOK_PATH,PREV_RUNS,MODELS]:
    if not os.path.exists(fpath):
        os.mkdir(fpath)

END_TOKEN           = "<|endoftext|>"

ALLOWABLE_CHAR      = string.ascii_lowercase + string.ascii_uppercase + "1234567890!@#$%^&*()~`':;{[}]_-+=<,>.?/}|\\ \n\t" + '"'

ENGL_DICT           = enchant.Dict("en_US")

LOWER               = False




#TRAINING SETTINGS 
UPDATE_EVERY_T                  = int(1*60)
SAMPLE_EVERY_T                  = 10*60
_LAST_UPDATE_T                  = time.time() 
_LAST_SAMPLE_T                  = time.time() - (3*60)
_SAVE_MODEL_EVERY               = 5000
_N_TOKENS                       = None
MAX_NORM                        = 1000
TOKENIZER                       = 'tokenizer'
CUR_STEP                        = 0 
TOT_STEP                        = 0
TOK_THRU                        = 0
MODEL                           = None
TOKENIZER                       = None
LOSS                            = None 
PROMPT                          = "<|endoftext|>"
UPDATE_FREQ                     = 10
SAVE_FREQ                       = 10_000


_LAST_UPDATE_T                  = time.time()
_LAST_SAMPLE_T                  = time.time()