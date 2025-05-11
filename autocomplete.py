import nltk
from nltk.corpus import words

# Download words dataset (only needed once)
nltk.download('words')

# Load word list
word_list = set(words.words())

def get_autocomplete_suggestions(text):
    """Return list of words that start with the input text."""
    if len(text) < 2:  # Avoid suggesting for single letters
        return []

    suggestions = [word for word in word_list if word.lower().startswith(text.lower())]
    return sorted(suggestions, key=lambda x: len(x))[:5]  # Return top 5 shortest words
