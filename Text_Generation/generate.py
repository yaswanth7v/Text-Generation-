import torch
from tokenizers import Tokenizer
from config import get_config
from model import *
import re

config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = Tokenizer.from_file("tokenizer.json")

pad_token = torch.tensor(tokenizer.token_to_id("[PAD]"), dtype=torch.int64)
seed_text = 'I could not help laughing at the ease with which he explained his process of deduction'
input_text = seed_text
next_words = 50

def preprocess(input_):

    seed_text = input_
    seed_text = re.sub(r'[^\w\s]', '', seed_text)

    sequence = tokenizer.encode(seed_text).ids

    # Add SOS token and pad the sequence
    sos_token = torch.tensor(tokenizer.token_to_id("[SOS]"), dtype=torch.int64)
    pad_token = torch.tensor(tokenizer.token_to_id("[PAD]"), dtype=torch.int64)
    
#---------------------- 
    if(sequence[0]==sos_token):
        sequence = sequence[1:]
    n = len(sequence) - config["seq_len"]
    if(n<0):
        n = 0
    sequence = sequence[n:]

    tokens = [sos_token]
    padding = [pad_token] * (config["seq_len"] - len(sequence)-1)
    tokens.extend(padding)
    tokens.extend(sequence)
#----------------------
    tokens = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0).to(device)  # Move to device
    return tokens


vocab_size = tokenizer.get_vocab_size()
model = build_transformer(vocab_size, config["seq_len"]).to(device)
model_path = 'model.pth'
if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Generate next words
for i in range(50):
    tokens = preprocess(input_text)
    src_mask = (tokens != pad_token).unsqueeze(-2)
    
    outputs = model.decode(src_mask.to(device), tokens.to(device))
    logits = model.project(outputs)
    logits = logits.squeeze(dim=-1)
    logits = logits.view(-1, logits.size(-1))

    _, next_word = torch.max(logits, dim=1)
    predicted_word = tokenizer.decode([next_word.item()])
    input_text += ' '+ predicted_word

result = input_text
print(result)