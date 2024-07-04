import torch
import string
import re
import json
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from config import get_config
from model import *

config = get_config()

def get_data():
    text = open('book.txt', 'r', encoding='utf-8').read()
    text = text.lower()
    text = text.replace('\n', ' ')
    sentences = text.split('.')
    sentences = sentences[50:]# to remove index content
    return sentences

def remove_punctuations(sentences):
    processed_sentences = []
  
    for sentence in sentences:
        if sentence!='':
            sentence = re.sub(r'[^\w\s]', '', sentence)
            processed_sentences.append(sentence)
    return processed_sentences

def tokenization(sentences, tokenizer):

    input_sequences = []
    sos_token = torch.tensor(tokenizer.token_to_id("[SOS]"), dtype=torch.int64)
    eos_token = torch.tensor(tokenizer.token_to_id("[EOS]"), dtype=torch.int64)
    pad_token = torch.tensor(tokenizer.token_to_id("[PAD]"), dtype=torch.int64)
    for sentence in sentences:
        n_gram_sequences = []
        sequence = tokenizer.encode(sentence).ids
        
        for i in range(1, len(sequence)):
            n_gram_sequence = sequence[:i+1]
            n_gram_sequences.append(n_gram_sequence)
        
        for sequence in n_gram_sequences:
            sequence = sequence[:config["seq_len"]]
            padding = [pad_token] * (config["seq_len"] - len(sequence))
            tokens = [sos_token] + padding 

            tokens.extend(sequence)
            tokens = torch.tensor(tokens, dtype=torch.int64)
            input_sequences.append(tokens)
            
    return input_sequences

def train_model(model, tokenizer, train_loader, val_loader, num_epochs, criterion, optimizer, device):

    train_losses = []
    val_losses = []
    for epoch in range(10, num_epochs+10):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for inputs in progress_bar:
            inputs = inputs.to(device)
            
            # Split inputs into source and target sequences
            source = inputs[:, :-1].to(device)  # Input sequence up to the second-to-last token
            target = inputs[:, -1].to(device)   # Target sequence from the first token onwards
            
            src_mask = (source != tokenizer.token_to_id("[PAD]")).unsqueeze(-2)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model.decode(src_mask, source)
            logits = model.project(outputs)
            logits = logits.squeeze(dim=-1)
            logits = logits.view(-1, logits.size(-1))
            
            target_flat = target.view(-1)  # shape: [batch_size * seq_len]

            # Calculate loss
            loss = criterion(logits, target_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():6.3f}"})
        
        model_save_path = f"model_text_generation_{epoch}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
        # Print epoch loss
        print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader)}")
        
        train_avg_loss = total_loss / len(train_loader)
        train_losses.append(train_avg_loss)
        # Validate the model after each epoch
        valid_loss = validate_model(model, tokenizer, val_loader, criterion, device)
        val_losses.append(valid_loss)
    return train_losses, val_losses

def validate_model(model, tokenizer, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.to(device)
            
            # Split inputs into source and target sequences
            source = inputs[:, :-1].to(device)  # Input sequence up to the second-to-last token
            target = inputs[:,-1].to(device)   # Target sequence from the first token onwards
            
            src_mask = (source != tokenizer.token_to_id("[PAD]")).unsqueeze(-2)
            
            # Forward pass
            outputs = model.decode(src_mask, source)
            logits = model.project(outputs)
            
            logits = outputs.squeeze(dim=-1)
            logits = logits.view(-1, logits.size(-1))

            target_flat = target.view(-1)  # shape: [batch_size * seq_len]
            loss = criterion(logits, target_flat)
        
            total_loss += loss.item()
    
    # Print validation loss
    print(f"Validation Loss: {total_loss / len(val_loader)}")
    val_loss = total_loss / len(val_loader)
    model.train()
    return val_loss

def main():

    sentences = get_data()
    processed_sentences = remove_punctuations(sentences)

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    trainer = WordLevelTrainer(special_tokens=["[PAD]", "[SOS]", "[EOS]", "[UNK]"])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(sentences, trainer)
    tokenizer.save("tokenizer.json")
    sequences = tokenization(processed_sentences, tokenizer)

    train_size = int(0.9 * len(sequences))
    val_size = len(sequences) - train_size
    train_dataset, val_dataset = random_split(sequences, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = tokenizer.get_vocab_size()

    model = build_transformer(vocab_size, config["seq_len"]).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    num_epochs = config["num_epochs"]
    train_losses, val_losses = train_model(model, tokenizer, train_loader, val_loader, num_epochs, criterion, optimizer, device)
    losses = {"train_losses": train_losses, "val_losses": val_losses}
    with open("losses.json", "w") as f:
        json.dump(losses, f)

if __name__=='__main__':
    main()
