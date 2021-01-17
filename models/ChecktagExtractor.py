import pandas as pd
import numpy as np
import os
os.system("pip install textblob")

from sklearn.base import BaseEstimator, TransformerMixin
from textblob import TextBlob

class ChecktagExtractor(BaseEstimator, TransformerMixin):
        def check_pos_tag(self,X):

        '''
        The function defines the transformer of tag check.
        Return: the frequency of 5 word classes: Noun,Pron,Verb,Adj,Adv.
        '''
            pos_family = {
        'noun' : ['NN','NNS','NNP','NNPS'],
        'pron' : ['PRP','PRP$','WP','WP$'],
        'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
        'adj' : ['JJ','JJR','JJS'],
        'adv' : ['RB','RBR','RBS','WRB']
        }
            pos_pool= ['NN','NNS','NNP','NNPS','PRP','PRP$','WP','WP$','VB','VBD','VBG','VBN','VBP','VBZ','JJ','JJR','JJS','RB','RBR','RBS','WRB']
            cnt = [[0,0,0,0,0]]*len(X)
            for i, x in enumerate(X):
                if(TextBlob(x)[0].isalpha()):
                    wikis = list(pd.DataFrame(TextBlob(x).tags)[1])
                    for wiki in wikis:
                        if wiki in pos_pool:
                            if wiki in pos_family['noun']:
                                cnt[i][0] += 1
                            elif wiki in pos_family['pron']:
                                cnt[i][1] += 1
                            elif wiki in pos_family['verb']:
                                cnt[i][2] += 1
                            elif wiki in pos_family['adj']:
                                cnt[i][3] += 1
                            else:
                                cnt[i][4] += 1
                    else:
                        continue
            return cnt

        def fit(self, X,Y):
            '''
            Model training function
            '''
                Y = None
                return self

        def transform(self, X):
            '''
            Model predicting function
            '''
            X_tagged = self.check_pos_tag(X)
            return pd.DataFrame(X_tagged)
