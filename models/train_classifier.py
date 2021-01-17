import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from ChecktagExtractor import *
from Tokenize import *

def load_data(database_filepath):
    '''
    The function loads the clean dataset from the sqlite database set in process_data.py.
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DR_MSG', engine)

    categ = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']
    X = df['message'].values
    Y = df[categ].values
    return X,Y,categ
    pass

def build_model():
    '''
    The function build model with
    machine pipeline which take in the message column as input and output
    classification results on the other 36 categories in the dataset.
    '''
    pipeline = Pipeline([
    ('features', FeatureUnion(
        [('nlp_pipeline', Pipeline(
            [('vect', CountVectorizer(tokenizer=tokenize)),
             ('tfidf', TfidfTransformer())
            ])),

        ('tag-chk', ChecktagExtractor())
    ])),

    # MultiOutputClassifier is for predicting multiple target variables.
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Using GridSearchCV to find optimal parameters for pipeline.
    parameters = {
        'features__nlp_pipeline__vect__ngram_range': ((1, 1), (1,2)),
        'features__nlp_pipeline__vect__max_df': (0.5, 0.75),
        'features__nlp_pipeline__tfidf__use_idf': (True, False),
    }

    cv = GridSearchCV(pipeline, param_grid = parameters,)

    return cv
    pass

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    The function report the f1 score, precision and recall for each output category of the dataset.
    '''

    Y_pred = model.predict(X_test)
    for i in range(36):
        category_name = category_names[i]
        labels = np.unique(Y_pred[:,i])
        confusion_mat = classification_report(Y_test[:,i], Y_pred[:,i],labels=labels)
        accuracy = (Y_pred[:,i] == Y_test[:,i]).mean()
        print("\nCategory:",category_name)
        print(" Labels:", labels)
        print(" Confusion Matrix:\n ", confusion_mat)
        print(" Accuracy:", accuracy)
    print("Best Parameters:", model.best_params_)
    pass

def save_model(model, model_filepath):
    '''
    The function export tained model as a pickle file
    '''
    s = pickle.dumps(model)
    with open(model_filepath,'wb+') as f: # mode is'wb+'，represents binary writen
        f.write(s)
    pass


def main():
    '''
    The main() function combines and executes all the above modules.
    '''

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
#Split data into train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

#Train pipeline
        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
