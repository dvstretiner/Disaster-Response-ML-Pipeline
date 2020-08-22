# Classifying Disaster Response Messages

## 1. Installations
This project uses Python version 3 and the following packages:
* numpy
* pandas
* nltk
* nltk --> ['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger']
* nltk --> pos_tag
* nltk.tokenize --> word_tokenize
* nltk.stem --> WordNetLemmatizer
* nltk.corpus --> stopwords
* sqlite3
* sklearn
* sklearn.base --> BaseEstimator, TransformerMixin
* sklearn.pipeline --> Pipeline, FeatureUnion
* sklearn.feature_extraction.text --> CountVectorizer, TfidfTransformer
* sklearn.metrics --> confusion_matrix, classification_report, f1_score, accuracy_score, precision_score, recall_score
* sklearn.model_selection --> train_test_split, GridSearchCV
* sklearn.ensemble --> RandomForestClassifier
* sklearn.multioutput--> MultiOutputClassifier
* pickle


## 2. Project Motivation
When disaster strikes, response agencies do not have time to parse through thousands of messages received from victims. Therefore, it is important to have a robust tool, which is able to compile messages from different sources, clean and classify them based on the need expressed: do people need water? shelter? medical help?

This project could be viewed as beta version of such tool. Although, it does not collect data from various sources, the project contains a web app with a dashboard showing stats on the messages received so far. It also includes a query field, where first responders could enter messages received and view which categories they would fall under.

*The following are possible categories:
* related
* request
* offer
* aid_related
* medical_help
* medical_products
* search_and_rescue
* security
* military
* child_alone
* water
* food
* shelter
* clothing
* money
* missing_people
* refugees
* death
* other_aid
* infrastructure_related
* transport
* buildings
* electricity
* tools
* hospitals
* shops
* aid_centers
* other_infrastructure
* weather_related
* floods
* storm
* fire
* earthquake
* cold
* other_weather
* direct_report

## 3. File Descriptions
Original data consists of two files: disaster_categories.csv & disaster_messages.csv. Categories were obtained based on the data in the messages file. 
Files exist within the following folder structure:

- app
	- template
		- master.html  *main page of web app
		- go.html  *classification result page of web app
	- run.py  *Flask file that runs app
	- utils.py *Custom transformer

- data
	- disaster_categories.csv  *data to process 
	- disaster_messages.csv  *data to process
	- process_data.py
	- DisasterResponse.db   *database to save clean data to

- models
	- train_classifier.py
	- utils.py *Custom transformer

The train_classifier.py file contains an ML pipeline consisting of three transformers and one RandomForestClassifier. Three transformers include CountVectorizer, TfidfTransformer and a custom transformer called NamedEntityChecker, which checks whether or not the message contains a named entity ('NNP' part of speech based on pos_tag). The pipeline is fine-tuned using GridSearchCV for optimal results.


## 4. How to Interact with this project
In order to output a web app with results, first the data must be cleaned and loaded into the database (ETL), then a classifier is run to produce a pickle file with the model (ML pipeline), finally a flask file containing code for the web app is run. 

1. **ETL pipeline**: data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
2. **ML pipeline**: models/train_classifier.py data/DisasterResponse.db, which would save the classifier.pkl file (it's about 180 MB)
3. **Flask file**: run.py

**master.html** contains three visualizations based on the existing disaster response data. In the text field, you can enter a new message and the model on the backend would output relevant classifications to the front end.

## 5. Results

The model produced the following *overall* results with respect to **test** data:
* *Accuracy*:  0.2624
* *Precision*:  0.8137
* *Recall*:  0.5162
* *F1*:  0.6316


**And the following category-level results**:

|Category|Precision|Recall|F1 Score|Support|
|-------|---------|------|--------|-------|
|related	|0.84	|0.94	|0.89	|5035|
|request	|0.81	|0.47	|0.59	|1133|
|offer	|0	|0	|0	|30|
|aid_related	|0.75	|0.63	|0.69	|2705|
|medical_help	|0.67	|0.08	|0.14	|496|
|medical_products	|0.79	|0.1	|0.18	|324|
|search_and_rescue	|0.6	|0.05	|0.1	|167|
|security	|0	|0	|0	|119|
|military	|0.5	|0.05	|0.09	|224|
|child_alone	|0	|0	|0	|0|
|water	|0.93	|0.34	|0.5	|432|
|food	|0.85	|0.62	|0.71	|731|
|shelter	|0.81	|0.41	|0.55	|586|
|clothing	|0.88	|0.14	|0.25	|98|
|money	|0.92	|0.07	|0.14	|150|
|missing_people	|0	|0	|0	|72|
|refugees	|0.83	|0.04	|0.08	|227|
|death	|0.73	|0.19	|0.3	|304|
|other_aid	|0.56	|0.04	|0.08	|884|
|infrastructure_related	|0.14	|0	|0	|445|
|transport	|0.66	|0.12	|0.21	|298|
|buildings	|0.7	|0.1	|0.17	|331|
|electricity	|0.5	|0.02	|0.04	|150|
|tools	|0	|0	|0	|37|
|hospitals	|0	|0	|0	|74|
|shops	|0	|0	|0	|31|
|aid_centers	|0	|0	|0	|80|
|other_infrastructure	|0	|0	|0	|300|
|weather_related	|0.85	|0.66	|0.74	|1833|
|floods	|0.85	|0.41	|0.56	|520|
|storm	|0.77	|0.46	|0.58	|594|
|fire	|1	|0.03	|0.06	|68|
|earthquake	|0.9	|0.75	|0.82	|631|
|cold	|0.82	|0.1	|0.17	|144|
|other_weather	|0.45	|0.03	|0.05	|357|
|direct_report	|0.74	|0.36	|0.48	|1247|

As you can see, the overall accuracy is not great: only ~26% of messages in test data were categorized correctly. However, do we really care about accuracy that much? Precision, recall and F1 score are all above 50%, which is not bad. Accuracy is a good metric when False Positives and False Negatives have equivalent costs. This is not the case in disaster situations - if anything, we want to be over-inclusive - tag as many categories as possible, because the cost of a false negative (e.g., a message about needing water not categorized as 'water') could be a human life. 

Based on this logic, Recall is our most important metric. Also called "Sensitivity" it is a ratio of True Positives to the Sum of True Positives and False Negatives. Recall measures classifier completeness: i.e., did we correctly identify all positive instances as such? 

Precision, also referred to as "Positive Predictive Value" evaluates classifier exactness: **TP / (TP + FP)**. It lets us ask the question: did we identify just the right number of positive instances, or did we overdo it? As I mentioned above, in the context of a disaster, recall is very important. However, we also don't want to mark all categories as positive for every message, that would overwhelm first responders. 

That's where the F1 score comes in. It combines both Precision and Recall with a harmonic mean. 

**F1 = 2 * ( (Precision * Recall) / (Precision + Recall) )**

This approach ensures that F1 is somewhere between Precision and Recall, but gives larger weight to smaller numbers. In our case, the overall F1 score is 0.6316, which is smaller than the traditional average of Precision and Recall, because it's dragged down by lower Recall. 

**Now, that we have our terminology in place, let's focus on the Classification Report by category**. F1 score is a good metric when we have imbalanced classes, i.e., when certain categories are under-represented in the data. In our case, the least common categories are Offer, Tools and Shops - each with less than 40 positive instances. Unfortunately, we were not able to capture these, as all metrics in the report are equal to zero. I should mention that Offer, Tools and Shops don't strike me as high emergency categories, so maybe that mis-classification is not to our detriment. 

What I personally care about the most is correctly classifying Medical Help, Water, Food and Shelter, since people's lives depend on a timely response in these categories. Medical help has not faired well with our model - producing a Recall score of only 0.08, which means we had a high number of False Negatives. Luckily, the other three critical categories have Recall higher than 0.3. A better tuned model or a different classifier could do the trick of improving scores for these important categories.


## 6. Licensing, Authors, Acknowledgements, etc.
The data used for this project came from Figure 8, a company that collects training data for AI. 
