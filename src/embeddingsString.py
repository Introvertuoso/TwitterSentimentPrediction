from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define your sentence
sentence = "This is a BERT model example.And we want to test the bert model output for different string with different lengths "

# Tokenize the sentence and obtain the attention mask as descriped in the paper and used by huggingface
inputs = tokenizer(sentence, return_tensors="pt")
tokens_tensor = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Predict hidden states features for each layer
with torch.no_grad():
    outputs = model(tokens_tensor, attention_mask=attention_mask)

# We only need the hidden states of the last layer (outputs[0]), and to get the sentence embedding we take the mean
sentence_embedding = outputs[0].mean(dim=1).squeeze()

# Print the embedding

print(f"{len(sentence_embedding)}")