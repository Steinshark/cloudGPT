import os 
import subprocess 
import sys
import json
import random
from utils import parse_wet_file, WebPage
import string 
from dataset import correct_by_dict, correct_to_ascii
from training import *
import pandas 

if not os.path.exists(DWNLD_PATH):
    with open(DWNLD_PATH,'w') as writefile:
        writefile.write(json.dumps([]))

def passes_vibe_check(text:str,threshhold=5):
    search_text     = text.lower()

    bad_count       = 0

    bad_words       = ["weight loss pill", 
                       "casino", "viagra", 
                       "testosterone booster", 
                       "miracle cure",
                       "hair loss treatment",
                       "brain booster",
                       "our top picks",
                       "terms and conditions",
                       "pussy",
                       "cunt",
                       "slut",
                       "fuck"]
    
    for word in bad_words:
        bad_count += search_text.count(word)
        if bad_count > threshhold:
            return False 
    
    return True

#Takes a path from a wet.paths.gz file and creates wet.paths 
def generate_urls(wet_fpath:str):

    #Extract the gz file to a temp path
    if not os.path.exists(f"{PATH}/temp"):
        os.mkdir(f"{PATH}/temp")
    temp_path       = f"{PATH}/temp/"
    subprocess.run(f'7z x "{wet_fpath}" "-o{temp_path}" -y')#"-o{ds_path}"
    

    #we just extracted to here, now build the URLS
    urls        = [url.strip() for url in open(f"{PATH}/temp/wet.paths",'r').readlines()]
    
    #Save to default fpath
    with open(URL_PATH,'w') as writefile:
        writefile.write(json.dumps(urls))

    return True


#Downloads a set number of files from the latest wet path
def download_files(n_files:int=128,lower=True,writefile_size=64,total_size=128*1024):
    #Establish fpath for tracking downloaded files

    prev_dwnld      = set(json.loads(open(DWNLD_PATH,'r').read()))
    
    paths           = set(json.loads(open(URL_PATH).read()))

    go_list         = list(paths - prev_dwnld)

    current_file        = os.path.join(f"{CRAWL_DB}",f"{random.randint(100_000_000_000,999_999_999_999)}.txt")
    writable_file       = open(current_file,"w",encoding='utf_8')

    current_size_MB     = 0 
    total_size_MB       = 0 


    #Tell us whats going on
    print(f"\n\nDownloading {n_files} files from {len(go_list)} total\nCreating {writefile_size}MB files")
    for url in go_list[:n_files]:

        url         = "https://data.commoncrawl.org/" + url
        downpath    = f"{PATH}/temp/temp.txt.gz"
        savepath    = f"{PATH}/temp/"

        #Download and unzip gunzip
        #subprocess.run(f"echo off")
        subprocess.run(f"curl {url} -o{downpath}") #Take out path name
        subprocess.run(f'7z x "{downpath}" "-o{savepath}" -y')


        #Open file 
        fileIO      = open(f"{savepath}/temp.txt",'r',encoding='utf_8')

        #Parse file 
        parsed_texts:list[WebPage] = parse_wet_file(fileIO,['eng'])
    
        for webpage in parsed_texts:
                
                #Generate cleaned page contents - only Ascii Chars
                text_addition   = webpage.contents + END_TOKEN
                text_addition   = "".join([c for c in text_addition if c in ALLOWABLE_CHAR])

                #Lower if set
                if lower:
                    text_addition = text_addition.lower()

                #How long was it?
                text_len        = len(text_addition.encode('utf-8'))

                #Write to current DB file
                writable_file.write(text_addition)


                #Measure stats
                current_size_MB += text_len/(1024*1024)
                total_size_MB   += text_len/(1024*1024)

                if current_size_MB > writefile_size:
                    print(f"current file size [{current_size_MB:.2f}MB] > {writefile_size}. Writing file")
                    writable_file.close()
                    current_size_MB     = 0 
                    current_file        = os.path.join(f"{CRAWL_DB}",f"{random.randint(100_000_000_000,999_999_999_999)}.txt")
                    writable_file       = open(current_file,"w",encoding='utf_8')

                if total_size_MB > total_size:
                    print(f"Crawl download complete: [{total_size_MB:.2f}MB]. exiting")
                    writable_file.close()

                    prev_dwnld.add(url)
                    with open(URL_PATH,'w') as writefile:
                        writefile.write(json.dumps(list(prev_dwnld)))
                    return 
        
        fileIO.close()
        os.remove(f"{savepath}/temp.txt")
        #add to prev downoaded
        prev_dwnld.add(url)
    
    with open(DWNLD_PATH,'w') as writefile:
        writefile.write(json.dumps(list(prev_dwnld)))


def clean_fineweb(writefile_size=32,min_score=.97):

    curfile     = os.path.join(FINEDB,str(random.randint(100_000_000,999_999_999))+".txt")
    curwrite    = open(curfile,'w',encoding='utf_8')

    for file in [os.path.join(FINE,fpath) for fpath in os.listdir(FINE) if ".parquet" in fpath]:

        data    = pandas.read_parquet(file,engine='pyarrow')

        for t,s,l in zip(data['text'],data['language_score'],data['language']):

            if l == 'en' and s > min_score and len(t) > 5_000 and passes_vibe_check(t):
                curwrite.write(t + END_TOKEN)

                if os.path.getsize(curfile) > (writefile_size * 1024 * 1024):
                    curwrite.close()
                    curfile     = os.path.join(FINEDB,str(random.randint(100_000_000,999_999_999))+".txt")
                    curwrite    = open(curfile,'w',encoding='utf_8')


if __name__ =='__main__':
    
    if len(sys.argv) > 1:
        fpath = sys.argv[1]
    else:
        fpath = 'C:/users/steinshark/downloads/wet.paths.gz'

    clean_fineweb(writefile_size=32,min_score=.9)
    #generate_urls(fpath)
    #download_files(8000,writefile_size=128)