import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
import threading
from model import LMSteinshark
from data import load_tokenizer
from utils import PROMPT_TOKEN, RESPONSE_TOKEN

class ModelTesterApp(tk.Tk):
    
    def __init__(self):
        super().__init__()
        self.title("SLM Model Tester")
        self.geometry("1280x800")
        self.model = None  # Placeholder for your loaded model
        self.tokenizer = None
        self._create_widgets()

        self.load_model(filepath="D:/production/model_weights.pth")
        self.load_tokenizer(tokenizer_dir="C:/gitrepos/cloudgpt/tokenizer")

        

    def _create_widgets(self):
        self.load_btn = ttk.Button(self, text="Load Model", command=self.load_model)
        self.load_btn.pack(pady=5, anchor='nw', padx=5)

        self.load_btn = ttk.Button(self, text="Load tokenizer", command=self.load_tokenizer)
        self.load_btn.pack(pady=5, anchor='nw', padx=5)

        prompt_label = ttk.Label(self, text="Prompt:")
        prompt_label.pack(anchor='nw', padx=5)
        self.prompt_text = ScrolledText(self, height=8)
        self.prompt_text.pack(fill='x', padx=5, pady=5)

        temp_frame = ttk.Frame(self)
        temp_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(temp_frame, text="Temperature:").pack(side='left')
        self.temp_var = tk.DoubleVar(value=1.0)
        self.temp_slider = ttk.Scale(temp_frame, from_=0.0, to=10.0, variable=self.temp_var, orient='horizontal')
        self.temp_slider.pack(side='left', fill='x', expand=True, padx=5)
        self.temp_display = ttk.Label(temp_frame, textvariable=self.temp_var, width=5)
        self.temp_display.pack(side='left')

        nt_frame = ttk.Frame(self)
        nt_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(nt_frame, text="n_tokens:").pack(side='left')
        self.n_tokens_var = tk.StringVar(value="128")
        self.n_tokens_entry = ttk.Entry(nt_frame, textvariable=self.n_tokens_var, width=10)
        self.n_tokens_entry.pack(side='left', padx=5)

        tk_frame = ttk.Frame(self)
        tk_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(tk_frame, text="top_k:").pack(side='left')
        self.top_k_var = tk.StringVar(value="40")
        self.top_k_entry = ttk.Entry(tk_frame, textvariable=self.top_k_var, width=10)
        self.top_k_entry.pack(side='left', padx=5)

        response_label = ttk.Label(self, text="Response:")
        response_label.pack(anchor='nw', padx=5)
        self.response_text = ScrolledText(self, height=15)
        self.response_text.pack(fill='both', expand=True, padx=5, pady=5)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill='x', padx=5, pady=5)
        self.clear_btn = ttk.Button(btn_frame, text="Clear", command=self.clear_texts)
        self.clear_btn.pack(side='right', padx=5)
        self.generate_btn = ttk.Button(btn_frame, text="Generate", command=self.start_generation_thread)
        self.generate_btn.pack(side='right', padx=5)

    def load_model(self,filepath=None):
        if filepath is None:
            filepath = filedialog.askopenfilename(
                title="Select Model Weights File",
                filetypes=[("PyTorch Weights", "*.pth"), ("All files", "*.*")]
            )
        
        if filepath:
            weights_path    = filepath
            config_path     = weights_path.replace("model_weights.pth","")[:-1]
            # Replace this stub with your model loading logic:
            # e.g. self.model = load_model_function(filepath)
            self.model = LMSteinshark.from_loadpoint(config_path,p_override=0).bfloat16().cuda().eval()
            self.response_text.insert(tk.END, f"Model loaded from: {config_path}\n")
    
    def load_tokenizer(self,tokenizer_dir=None):
        if tokenizer_dir is None:
            tokenizer_dir       = filedialog.askdirectory(title="Select Tokenizer Path")
        self.tokenizer      = load_tokenizer(tokenizer_dir)
        self.response_text.insert(tk.END,'Loaded tokenizer.\n')

    def clear_texts(self):
        #self.prompt_text.delete('1.0', tk.END)
        self.response_text.delete('1.0', tk.END)

    def start_generation_thread(self):
        if not self.model:
            self.response_text.insert(tk.END, "Please load a model first.\n")
            return
        if not self.tokenizer:
            self.response_text.insert(tk.END,"Please load a tokenizer.\n")
        # Disable generate button during generation
        self.generate_btn.config(state='disabled')
        threading.Thread(target=self.generate_response, daemon=True).start()

    def generate_response(self):
        prompt = self.tokenizer.encode(f"{PROMPT_TOKEN}{self.prompt_text.get('1.0', tk.END).strip()}{PROMPT_TOKEN}{RESPONSE_TOKEN}").ids
        try:
            temperature = float(self.temp_var.get())
            n_tokens = int(self.n_tokens_var.get())
            top_k = int(self.top_k_var.get())
        except ValueError:
            self.response_text.insert(tk.END, "Invalid parameter(s). Please check temperature, n_tokens, and top_k.\n")
            self.generate_btn.config(state='normal')
            return

        # Clear previous response
        self.response_text.delete('1.0', tk.END)

        # Call your model's token_streamer and append tokens as they come
        try:
            for token in self.model.token_streamer(prompt, self.tokenizer, n_tokens, temperature, top_k):
                # Insert token in the GUI thread
                self.response_text.after(0, self._append_token, token)
        except Exception as e:
            self.response_text.after(0, self.response_text.insert, tk.END, f"\n[Error during generation: {e}]")

        # Re-enable generate button after done
        self.response_text.after(0, self.generate_btn.config, {'state': 'normal'})

    def _append_token(self, token):
        self.response_text.insert(tk.END, token)
        self.response_text.see(tk.END)  # Scroll to bottom

# Dummy model class for demonstration, replace with your actual model object
class DummyModel:
    def token_streamer(self, prompt, temperature, n_tokens, top_k):
        # Dummy token generator example
        import time
        sample_response = "This is a dummy streamed response to your prompt. "
        for ch in sample_response:
            time.sleep(0.05)  # simulate generation delay
            yield ch

if __name__ == "__main__":
    app = ModelTesterApp()
    app.mainloop()
