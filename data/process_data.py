import numpy as np
import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    file paths to messages and categories csv files

    OUTPUT
    a merged dataframe consisting of the messages and categories datasets
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='inner')

    return(df)

def clean_data(df):
    '''
    INPUT
    merged dataframe from the load step

    OUTPUT
    clean dataframe with separate columns for each category
    category values in each column are binary: 0 or 1
    '''

    #Make categories a separate df, create columns for each separate category
    categories = pd.DataFrame(df['categories'].str.split(";", expand=True))

    #Make category names column names in the categories dataframe
    row = categories.loc[0,:]
    category_colnames = [category.split('-')[0] for category in row]
    categories.columns = category_colnames

    #Convert category values in the dataframe to numeric 0 or 1
    for column in categories:
        categories[column] = [category[1] for category in categories[column].str.split('-')]
        categories[column] = pd.to_numeric(categories[column])

    #Replace the original 'categories' column in df with the new category columns
    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1)
    
    #Ensure that 'Related' column has only 1's and 0's
    df.related.replace(2,1,inplace=True)

    #Remove duplicate rows
    df = df.drop_duplicates()

    return(df)

def save_data(df, database_filename):
    '''
    INPUT
    dataframe from the clean step

    OUTPUT
    database table created from the clean dataframe
    '''
    #Connect to the database & write the table
    conn = sqlite3.connect(database_filename)
    df.to_sql('DisasterMessages', con = conn, if_exists='replace', index=False)

    #Commit & close connection
    conn.commit()
    conn.close()

    return None

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

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
