 Steinshark's LLM

## Current Work
Here you'll find my LLM project - a 1B parameter GPT-2 style model. This model was trained on mainly English and Python Code, uses RoPE, has a vocab of 32768, and is a project I love to spend late nights perfecting! 
Currently we're on the Finetuning phase. I'm using ELI5, Baize, Natural Questions, code Alpaca, and a Reddit dataset as an unsupervised dataset. Next, I'll be collecting human feedback on response pairs to try to 
perform some RLHF - stay tuned, I hope I can actually get people to participate.

## Usage 
The model is found in model.py. I have a training script in train.py, and useful environmental parameters in environment.py
The datset expected by the model is a folder path containing tokens saved in a .npy format. I'll try to find a way to publish that later, currently 
I have around 90GB of Wikipedia, Fineweb, The Stack (Python only), and some other stuff...

## Feedback
I'm new at this. Let me know if anything can be improved, made cleaner, or whatnot. Thanks for checking this out!