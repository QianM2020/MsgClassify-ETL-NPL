import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    #messages.head()
    categories = pd.read_csv(categories_filepath)
    #categories.head()
    df = messages.merge(categories,how = 'outer', on =('id'))
    #df.head()
    return df
    pass


def clean_data(df):
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[1]
    # use this 1st row to extract a list of new column names for categories.
    category_colnames = row.replace('-0', '', regex=True).replace('-1', '', regex=True)
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.get(-1)
        categories[column] = pd.to_numeric(categories[column])        
    # drop and concatenate the original dataframe with the new `categories` dataframe
    df.drop(['categories'],axis = 1, inplace = True)
    df = pd.concat([df,categories],axis = 1)
    #remove duplicates
    df.drop_duplicates(inplace = True)
    return df
    pass


def save_data(df, database_filename): 
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DR_MSG', engine, index=False)
    pass  


def main():
    if len(sys.argv) == 4:
        #print('sys.argv[1:] is :', sys.argv[1:])

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        #print('df is:', df.head())
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()