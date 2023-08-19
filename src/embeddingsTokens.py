from transformers import BertModel, BertTokenizer
import torch

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Define your list of tokens
tokens = ["hello", "world", "bert", "example"]

# Encode the tokens to their respective IDs used by Huggingface models
input_ids = torch.tensor([tokenizer.encode(tokens, add_special_tokens=True)])

# Generate the embeddings for the tokens
with torch.no_grad():
    outputs = model(input_ids)

# Extract the last hidden states (representations for each token)
last_hidden_states = outputs[0]

# The `last_hidden_states` tensor has shape [batch, sequence_length, model_dimension]
# Here `sequence_length` includes the special tokens `[CLS]` and `[SEP]`,
# so we slice the tensor along the second dimension to obtain our original tokens representations
token_embeddings = last_hidden_states[0, 1:-1]

for token in token_embeddings:
  print(f"{len(token)}")
print(token_embeddings)
