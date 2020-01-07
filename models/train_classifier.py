import sys
import pandas as pd
import numpy as np
import seaborn as sns
import re
import nltk
import pickle

from sqlalchemy import create_engine

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def load_data(database_filepath):
    '''Loads data table from SQLite db and returns X, Y and category names'''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(database_filepath, engine)
    
    ## Remove column original/child_alone as it doesn't add any value. Child alone has zero variance
    df = df.drop(columns = ['original','child_alone'])
    
    ##Dropping whole rows of NaNs
    df = df.dropna(axis = 0)
    
    X = df['message']
    Y = df.iloc[:,3:]
    category_names = list(Y.columns)
    
    return (X, Y, category_names)


def tokenize(text):
    # Remove urls
    url_regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    text = re.sub(url_regex, "url_placeholder", text)
    
    # Normalize case
    text = text.lower()
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # Lemmatize remove stop words and stem
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''Transformer class to check if the first word is a verb'''

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) == 0:
                return False
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)



def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(LogisticRegression(class_weight = 'balanced')))
    ])
    return pipeline

def custom_recall_scorer(y_true, y_pred):
    '''Returns a mean recall score of each of the categories. This scorer is used while performing GridSearch'''
    recall_scores = []
    for i in range((y_true.shape[1])):
        recall_scores.append(recall_score(y_true = y_true.iloc[:,i], y_pred = y_pred[:,i]))
    return np.mean(recall_scores)

def grid_search_for_params(X_train, y_train, model,  custom_scorer):
    '''Run a grid search on select params to return a model with optimized params according to custom scorer for the pipeline'''
    #Build a custom scorer
    scorer = make_scorer(custom_scorer)
    
    #Parameter for grid_search
    parameters = {'clf__estimator__C':[1, 10, 100]}
    
    cv = GridSearchCV(model, parameters, cv=3, scoring=scorer)

    cv.fit(X_train, y_train)
    
    return cv

def print_classification_report(y_pred, y_test, colnames):
    '''Prints a classification report for each of the columns'''
    for i in range(y_test.shape[1]):
        print(colnames[i])
        print(classification_report(y_true = y_test.iloc[:,i], y_pred = y_pred[:,i]))
    return None


def evaluate_model(model, X_test, Y_test, category_names):
    X_test_mult_op = model.predict(X_test)
    print_classification_report(X_test_mult_op, Y_test)
    return None


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
#         model.fit(X_train, Y_train)
        besst_model = grid_search_for_params(X_train, Y_train, model, custom_recall_scorer)
        
        print('Evaluating model...')
        evaluate_model(best_model, X_test, Y_test, category_names)

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