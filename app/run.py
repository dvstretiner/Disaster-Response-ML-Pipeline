import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine
from utils import NamedEntityChecker
import plotly.express as px



app = Flask(__name__)

def tokenize(text):
    
    '''
    INPUT
    Text of the disaster message

    OUTPUT
    Cleaned tokens: lower case, without punctuation, lemmatized
    '''
    
    #Remove punctuation, convert to lower case. Instantiate a Lemmatizer
    text = re.sub(r"[^a-zA-z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    #Lemmatize tokens, strip of extra spaces, return clean tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

categ_dict = {}

def top_n_categories(data, n):
    
    '''
    INPUT
    dataframe, top number to select
    
    OUTPUT
    x_values: category names for top N categories
    y_values: category counts for top N categories
    
    categories exclude the first column 'related' b/c it is the most prevalent
    '''
    #Drop any duplicate rows
    data = data.drop_duplicates()
    
    #Get categories and counts, excluding the 'related' group
    categories = data.columns[5:]
    counts = [df[category].sum() for category in categories]
    
    #Create a dictionary of key, value pairs for categories & counts
    for key in categories:
        for value in counts:
            categ_dict[key] = value
            counts.remove(value)
            break
            
    #Obtain a sorted list in descending order, grab the top n
    sorted_categories_top_n = sorted(categ_dict.items(), key=lambda x: x[1], reverse=True)[:n]
    
    #Obtain x and y values
    top_n_categories = [str.replace(x[0], '_', ' ').title() for x in sorted_categories_top_n]
    top_n_counts = [y[1] for y in sorted_categories_top_n]
    
    return top_n_categories, top_n_counts

# load data from the database
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterMessages', engine)

# load model from the pickle file
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data for a bar chart of genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extract data for a bar chart of top 10 categories present in data (excluding 'related')
    top_10_categories, top_10_counts = top_n_categories(df, 10)
    
    # extract data for a histogram of word counts in the disaster messages
    word_counts = [len(tokenize(text)) for text in df['message'].drop_duplicates()]
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }, 
        
        {
            'data': [
                Histogram(
                    x=word_counts,
                    xbins=dict( # bins used for histogram
                          start=0,
                          end=200,
                          size=10                        
                            ),
                   marker=dict(
                        color='#468499'
                   )
                    
                )
            ],
            
            'layout': {
                'title': 'Distribution of Word Counts in Disaster Messages',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Word Count Bins",
                    
                },
                
                'plot_bgcolor': '#DCDCDC'
                
            }
        }, 
        

        {
            'data': [
                Bar(
                    x=top_10_categories,
                    y=top_10_counts,
                    marker =  {
                        'color':  'rgb(142,124,195)'
                            }
                )
                ],

            'layout': {
                'title': 'Top 10 Disaster Message Categories',
                'yaxis': {
                    'title': "Number of Positive Instances"
                },
                'xaxis': {
                    'title': "Top 10 Categories"
                }, 
                
                'plot_bgcolor': '#DCDCDC'
                
            }
        }
               
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()