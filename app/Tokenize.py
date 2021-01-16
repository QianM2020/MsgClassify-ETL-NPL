import re
import pandas as pd
import numpy as np

import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize  
from nltk.tokenize import sent_tokenize
nltk.download('wordnet') 
from nltk.stem.wordnet import WordNetLemmatizer 

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text) 
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() # remove capital letters and space
        clean_tokens.append(clean_tok)
    return clean_tokens
    pass