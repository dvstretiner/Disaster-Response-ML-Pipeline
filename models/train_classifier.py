import sys
import re
import numpy as np
import pandas as pd
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
from utils import NamedEntityChecker

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import sqlite3
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    INPUT
    file path to the database containing the DisasterMessages data

    OUTPUT
    X - Series of disaster messages
    Y - Dataframe of categories encoded as 0 or 1
    category_names - labels for all categories
    '''

    #Load table from the database
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterMessages", con=conn)
    conn.commit()
    conn.close()

    #Ensure that only 0s and 1s are present (i.e., no 2's)
    for col in df.columns[4:]:
        df.loc[(df[col]>1), col]=1

    #Obtain X,Y and category labels
    X =  df['message']
    Y =  df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = df.columns[4:]

    return X, Y, category_names

#Add more stopwords to the list based on observations in the data
my_stopwords = ['thank', 'thanks', 'hello', 'god', 'bless', 'im', 'u', 'please', 'ou']
all_stopwords = stopwords.words("english")
for word in my_stopwords:
    all_stopwords.append(word)

def tokenize(text):
    '''
    INPUT
    Text of the disaster message

    OUTPUT
    Cleaned tokens: lower case, without punctuation & stopwords, lemmatized
    '''

    #Normalize text: make lower-case, remove punctuation
    text = re.sub(r"[^a-zA-z0-9]", " ", text.lower())
    tokens = word_tokenize(text)

    #Remove Stop Words & extra spaces
    tokens_no_stops = [tok.strip() for tok in tokens if tok.strip() not in all_stopwords]

    #Lemmatize
    tokens_lemmatized = [WordNetLemmatizer().lemmatize(tok) for tok in tokens_no_stops]

    return tokens_lemmatized


def build_model():
    '''
    INPUT
    None

    OUTPUT
    A model consisting of a pipeline with text transfomers and a classifier.
    The classifier is tuned to optimal parameters via GridSearch.
    '''

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('named_entities', NamedEntityChecker())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state = 100)))
    ])

    parameters = {
        'clf__estimator__n_estimators': [10,20]
    }

    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs = 3, verbose = 10, cv = 2)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
    model fitted on X_train and Y_train
    X_test, Y_test and category labels

    OUTPUT
    Category-level classification report
    Overall (dataset level) accuracy, precision and F1 Score
    '''

    #Predict categories from unsees data (X Test)
    Y_pred = model.predict(X_test)

    #Obtain classification report at the category level showing precision, recall and F1 Score
    category_class_report = classification_report(np.float64(Y_test), Y_pred, target_names=category_names)

    #Obtain data-set level metrics: accuracy, precision, recall and F1

    #Note: precision, recall and F1 use the "micro" average, which
    #takes data-set level totals of True Positives, False Positives and False Negatives
    overall_accuracy = accuracy_score(np.float64(Y_test), Y_pred)
    overall_precision =  precision_score(np.float64(Y_test), Y_pred, average='micro')
    overall_recall =  recall_score(np.float64(Y_test), Y_pred, average='micro')
    overall_f1 =  f1_score(np.float64(Y_test), Y_pred, average='micro')

    print('Category-Level Classification Report: ', category_class_report)
    print('Overall Accuracy: ', overall_accuracy)
    print('Overall Precision: ', overall_precision)
    print('Overall F1 Score: ', overall_f1)

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

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
