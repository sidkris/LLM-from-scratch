import torch 
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2Tokenizer
import warnings
warnings.filterwarnings("ignore")

class SimpleTransformer(nn.Module):

    def __init__(self, vocab_size, input_dim, n_heads, num_layers):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.transformer = nn.Transformer(input_dim, n_heads, num_layers)
        self.fc_output = nn.Linear(input_dim, vocab_size)

    
    def forward(self, x):

        x = self.embedding(x)
        x = self.transformer(x, x)
        return self.fc_output(x)
    


dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_texts = dataset["train"]["text"]

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokens = tokenizer(train_texts, return_tensors = "pt", padding = True, truncation = True)

vocab_size = tokenizer.vocab_size
model = SimpleTransformer(vocab_size = vocab_size, input_dim = 512, n_heads = 8, num_layers = 6)


# SPLIT THE TOKENIZED DATA INTO SMALLER BATCHES
batch_size = 8
input_ids = tokens["input_ids"].split(batch_size)


# TRAINING
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

for epoch in range(10):

    for batch in input_ids:

        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs.view(-1, vocab_size), batch.view(-1))
        loss.backward()
        optimizer.step()
    
    print(f"EPOCH : {epoch} | LOSS : {loss.item}")



# GENERATE TEXT
def generate_text(model, tokenizer, prompt, max_length = 100):

    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors = "pt")

    for _ in range(max_length):

        outputs = model(input_ids)
        predictions = outputs[:, -1, :].argmax(dim = -1)
        input_ids = torch.cat([input_ids, predictions.unsqueeze(0)], dim = 1)

    return tokenizer.decode(input_ids[0], skip_special_tokens = True)


print(generate_text(model, tokenizer, "Once upon a time"))


