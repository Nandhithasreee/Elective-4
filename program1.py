import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

nlp = spacy.load("en_core_web_sm")

text = "Apple Inc. was founded by Steve Jobs in 1976. It is headquartered in Cupertino, California."

tokens = word_tokenize(text)
print("\nTokens:")
print(tokens)

stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.isalnum() and word.lower() not in stop_words]

print("\nAfter Stop Word Removal:")
print(filtered_tokens)

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]

print("\nStemming:")
print(stemmed_words)

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]

print("\nLemmatization:")
print(lemmatized_words)

pos_tags = nltk.pos_tag(tokens)

print("\nPOS Tagging:")
print(pos_tags)

doc = nlp(text)
print("\nNamed Entities:")
for ent in doc.ents:
    print(ent.text, "->", ent.label_)

word_freq = Counter(filtered_tokens)

print("\nWord Frequency:")
for word, freq in word_freq.items():
    print(word, ":", freq)