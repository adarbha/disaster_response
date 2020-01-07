import sys
import pandas as pd

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''Load the two dataframes and get a merged dataset. 
    Categories dataset is wrangled to split data values as column headers'''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #Merge messages and categories
    df = pd.merge(messages, categories, on = ['id'], how = 'outer')
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    
    # select the first row of the categories dataframe to use this as colnames 
    row = categories.iloc[1,:]
    category_colnames = row.apply(lambda x:x[:-2])
    
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        categories[column] = categories[column].apply(lambda x:str(x)[-1])
        categories[column] = pd.to_numeric(categories[column])
        
       
    # Related column has a third key called 2. As no information regarding this is provided, removing all such classes
    categories = categories[categories['related'] != 2]
    
    # drop the original categories column from `df` and concat with new categories df
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis = 1)
    
    return df

    

def clean_data(df):
    '''Remove duplicates. '''
    # drop duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    '''Saving to sqlite database DisasterResponse.db'''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('msg_cat', engine, index=False)


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