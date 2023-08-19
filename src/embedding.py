from sentence_transformers import SentenceTransformer


def base_embedder(words_str, name = 'bert-base-nli-mean-tokens'):
    # name = 'stsb-distilbert-base'
    # name = 'average_word_embeddings_komninos'

    model = SentenceTransformer(name)

    sentences = list(words_str)
    sentence_embeddings = model.encode(sentences)

    return sentence_embeddings