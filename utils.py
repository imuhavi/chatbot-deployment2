from nltk_utils import stem
import numpy as np

def bag_of_words(tokenized_sentence, all_words):
    """
    Create a bag-of-words vector representation of a tokenized sentence.

    Args:
        tokenized_sentence (list): List of tokenized words in a sentence.
        all_words (list): List of all unique words in the corpus.

    Returns:
        bag (numpy.ndarray): Bag-of-words vector representation.
    """
    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] = 1.0
    return bag
