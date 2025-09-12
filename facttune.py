import random 
import json
import mwparserfromhell
import bz2
import gzip
import os
import html
from datasets import load_dataset
import xml.etree.ElementTree as ET
from utils import PROMPT_TOKEN,RESPONSE_TOKEN

# ---------------------------
# Helper functions
# ---------------------------
wiki_dump = "F:/data/enwiki-20250401-pages-articles-multistream.xml.bz2"

ns = {"mw": "http://www.mediawiki.org/xml/export-0.11/"}

prompt_templates = [
    "Provide factual information about {subject}.",
    "What are the key details about {subject}?",
    "Summarize important facts regarding {subject}.",
    "Give a concise explanation of {subject}.",
    "Describe the main points about {subject}.",
    "What is known about {subject}?",
    "Provide an overview of {subject}.",
    "List the relevant facts related to {subject}.",
    "Explain the significance of {subject}.",
    "Give a factual summary of {subject}.",
    "What are the notable aspects of {subject}?",
    "Share accurate information concerning {subject}.",
    "What details are known about {subject}?",
    "Provide a short factual description of {subject}.",
    "Summarize the topic {subject} in a factual manner.",
    "What does research say about {subject}?",
    "Present key points about {subject}.",
    "Offer an objective overview of {subject}.",
    "Explain {subject} in a factual, unbiased way.",
    "Give a clear and accurate summary of {subject}.",
]


def clean_wiki_text(wiki_markup):
    """Convert wiki markup to clean text."""
    wikicode = mwparserfromhell.parse(wiki_markup)
    text = wikicode.strip_code()
    text = html.unescape(text)
    # Remove excessive newlines
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return text

def yield_wikipedia_articles(path):
    open_func = bz2.open if path.endswith(".bz2") else open
    with open_func(path, "rb") as f:
        context = ET.iterparse(f, events=("end",))
        for event, elem in context:
            if elem.tag == "{http://www.mediawiki.org/xml/export-0.11/}page":
                title = elem.find("./mw:title", ns)
                text  = elem.find("./mw:revision/mw:text", ns)
                if title is not None and text is not None:
                    yield title.text, clean_wiki_text(text.text)
                elem.clear()

def generate_qa_from_wiki(title, text):
    """Yield simple QA pairs using headings as pseudo-questions."""
    lines = text.split("\n")
    for line in lines:
        if len(line.strip()) > 40:  # skip very short lines
            prompt = random.sample(prompt_templates).replace('{subject}',title)
            response = line.strip()
            yield f"{PROMPT_TOKEN}{prompt}{RESPONSE_TOKEN}{response}"

# ---------------------------
# Pipeline
# ---------------------------

def factual_dataset_pipeline(wiki_dump_path, output_file, limit=None):
    with open(output_file, "w", encoding="utf-8") as out_f:
        for title, text in yield_wikipedia_articles(wiki_dump_path):
            for qa in generate_qa_from_wiki(title, text):
                input(qa)
                out_f.write(json.dumps(qa) + "\n")

# ---------------------------
# Optional QA dataset merging
# ---------------------------

def merge_nq_dataset(output_file):
    # Using Huggingface Natural Questions dataset
    nq = load_dataset("natural_questions", "default")
    with open(output_file, "a", encoding="utf-8") as out_f:
        for split in ["train", "validation"]:
            for example in nq[split]:
                question = example["question_text"]
                answer = " ".join([a["text"] for a in example["annotations"][0]["short_answers"]])
                if len(answer.strip()) > 0:
                    qa = {"prompt": question, "response": answer.strip()}
                    out_f.write(json.dumps(qa) + "\n")


wiki_dump = "Steinpc/s/nlp/data/enwiki-20250401-pages-articles-multistream.xml.bz2"
output_file = "Steinpc/s/nlp/data/factual_dataset.jsonl"

# Generate Wikipedia-based QA
factual_dataset_pipeline(wiki_dump, output_file, limit=100_000)  # limit optional

# Merge Natural Questions
merge_nq_dataset(output_file)