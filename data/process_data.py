import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    The function loads data from messages.csv,categories.csv,
    and merge them into a dataframe.
    '''
    messages = pd.read_csv(messages_filepath)
    #messages.head()
    categories = pd.read_csv(categories_filepath)
    #categories.head()
    df = messages.merge(categories,how = 'outer', on =('id'))
    #df.head()
    return df
    pass


def clean_data(df):
    '''
    The function splits categories into separate category columns,
    converts category values to just numbers 0 or 1, and removes duplicates.
    '''
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
    '''
    The function saves the clean dataset into an sqlite database.
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    #use the if_exists='replace' option in to_sql() when saving the data so that
    # the pipeline can be executed several times without having a ValueError.
    df.to_sql('DR_MSG', engine, index=False,if_exists='replace')
    pass


def main():
    '''
    The main() function combines and executes all the above modules.
    '''
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
