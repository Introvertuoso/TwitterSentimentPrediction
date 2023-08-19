from math import floor
import pandas as pd
import validators
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import spacy
from typing import List

# import contextualSpellCheck


# Add the stopwords dataset to nltk
nltk.download("stopwords")
# TODO: Document
spacy_tagger = spacy.load("en_core_web_sm", disable=["parser", "ner"])
# contextualSpellCheck.add_to_pipe(spacy_tagger)

# Punctuation markers we are interested in
relevant_punctuation = ['?', '!']
# Configure which language to use for the stopwords
stopwords = set(stopwords.words("english"))


# TODO: REWRITE TO INPUT=SENTENCE AND OUTPUT=SENTENCE


def replace_url_word(word, replace_with_tag=True):
    if validators.url(word):
        return "-URL-" if replace_with_tag else ""

    return word


def process_urls(paragraph, replace_with_tag=True):
    """
    Goes through each word in the paragraph and replaces it with a URL tag (by default),
    or an empty string.
    """
    return " ".join([replace_url_word(word) for word in paragraph.split()])


def is_numeric(string):
    """
    Returns True if the input string only contains numeric characters, False otherwise.
    Also returns False if the input is not a string.
    """
    # Method only works with str
    return str.isnumeric(string) if string is str else False


def is_alphabetic(string):
    """
    Returns True if the string is only made up of alphabetic characters, False otherwise.
    Also returns False if the input is not a string.
    """
    # Method only works with str
    return str.isalpha(string) if string is str else False


def is_tag(word):
    """
    It's a tag if the word both begins and ends with a dash (-)
    """
    return word[0] == '-' and word[-1] == '-'


def is_all_caps(word):
    """
    Returns True if all the letters in the word are capitalized,
    and False if not. Also returns False if the input is not a string.
    """
    # Method only works with str
    return all(map(str.isupper, word)) if word is str else False


def map_letter_casing_word(word: str):
    """
    Function for use in map-style operations that maps words to their case-adjusted representation, where words written in all caps will be represented as "ALL_CAPS_{word}",
    and all other words are returned in their lowercase representation. If the input is None (denoting a removed element), None is simply returned again.
    """
    if word is None:
        return None

    if is_all_caps(word):
        # Distinguish emotional emphasis when words are written in all-caps
        return "-ALL_CAPS-" + word

    else:
        # Will ignore incompatible words and symbols, such as punctuation marks
        return word.lower()


def map_letter_casing(paragraph: str):
    """
    Wrapper function that executes map_letter_casing_word for each word in the paragraph
    """
    # Splits string using whitespace
    word_list = paragraph.split()
    word_list = [map_letter_casing_word(word) for word in word_list]

    return " ".join(word_list)


def contains_only_relevant_punctuation(word):
    """
    Returns True if all the characters in word are either exclamation marks or question marks,
    and False if not.
    """
    return all(map(lambda c: c in relevant_punctuation, word))


"""
"""


# TODO: Document
def string_consists_only_of_char(string, char):
    return all(x == char for x in string)


# TODO: Add ellipses?
# TODO: Add support for exclaimed question?
def map_punctuation_markers(word, in_readable_form=False):
    """
    Maps punctuation strings and alphabetic words to tags representing interesting punctuation strings and the
    same alphabetic words. Non-interesting punctuation strings are ignored.
    This function assumes that all punctuation marks appear as isolated elements, and that
    numbers and URLS are absent.

    By default, it replaces the punctuation marks and punctuation clusters with tags. However, it can also be
    replaced with

    The default substitution value for each is the same as the key name. Use None to ignore.
    """
    if is_alphabetic(word) or is_tag(word):
        # Add without further processing
        return word

    # If the word only contains ! and ? (subject to change)
    if contains_only_relevant_punctuation(word):
        if len(word) == 1:
            if string_consists_only_of_char('?'):
                return '?' if in_readable_form else "-QUESTION_MARK-"

            if string_consists_only_of_char('!'):
                return '!' if in_readable_form else "-EXCLAMATION_MARK-"

        # If len(word) > 1
        if string_consists_only_of_char('?'):
            return "??" if in_readable_form else "-MULTI_QUESTION_MARK-"

        # If only multiple question marks
        if string_consists_only_of_char('!'):
            return "!!" if in_readable_form else "-MULTI_EXCLAMATION_MARK-"

    # Other strings are ignored/removed
    return None


def process_stopword(word):
    """
    Returns an empty string if the word is in NLTK's stopword list. Otherwise, it returns the input unchanged.
    """
    return "" if word in stopwords else word


def remove_stopwords(sentence):
    """
    Returns a new sentence where each stopword is removed
    """
    return " ".join([token.lemma_ for token in sentence.split()])


# TODO: Implement???
# How to do this since it can't work on only one element individually
def map_negation_to_antonym():
    pass


# TODO: Handle pronoun tag (check if word begins and ends with a dash)
def map_sentence_to_lemmatized_form(paragraph):
    """
    Returns the paragraph in lemmatized form. Extracts the token.lemma_ for each in the paragraph,
    as returned by spaCy, and joins them to a new paragraph before returning.
    """
    return " ".join([token.lemma_ for token in spacy_tagger(paragraph)])


def remove_infrequent_words(list_of_bag_of_words: List[List[str]], treshold_percentage: float) -> List[List[str]]:
    """
    """
    """
    # TODO: Document
    def map_to_correct_spelling(paragraph):
        return spacy_tagger(paragraph)._.outcome_spellCheck
    """
    frequency_map = {}

    # Count the number of occurences of each word
    for tweet in list_of_bag_of_words:
        for word in tweet:
            if word in frequency_map.keys():
                frequency_map[word] += 1

            else:
                frequency_map[word] = 1

    frequency_list = list(frequency_map.items())
    # word[1] is the number of occurences
    frequency_list.sort(key=lambda word: word[1], reverse=False)
    # Round down the amount of deleted words in case of decimal values
    number_of_deleted_words = floor((len(frequency_list) / 100) * treshold_percentage)
    # Delete the least frequent words
    words_to_delete = frequency_list[:number_of_deleted_words]

    filtered_tweets = []

    for tweet in list_of_bag_of_words:
        # Filter out the words to delete, and create a new list
        filtered_tweets.append(list(filter(lambda word: word not in words_to_delete, tweet)))

    return filtered_tweets
