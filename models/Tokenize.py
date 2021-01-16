import re
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize  
from nltk.tokenize import sent_tokenize
nltk.download('wordnet') 
from nltk.stem.wordnet import WordNetLemmatizer 

def tokenize(text):
    # Remove url
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text) 
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")   
    #tokenize and lemetiz the text    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer() 
    
    # remove capital letters and space
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() 
        clean_tokens.append(clean_tok)
    return clean_tokens
 