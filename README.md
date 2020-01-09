# Disaster Response Pipeline Project


## Table of Contents
1. ### Installation
    This code is run on Python 3.6.x. Please install all the python dependencies using **requirements.txt**.
2. ### Project Motivation
	 Analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. Using real messages that were sent during disaster events, goal is to create a machine learning pipeline to categorize these events into 36 categories. Categories are extracted from input data. An emergency worker can input a new message and get classification results with respective probabilty of the category being true, for selected categories.
    
3. ### Files and their description
Below is the directory structure</br>
.</br>
├── app</br>
│   ├── run.py</br>
│   └── templates</br>
│       ├── go.html</br>
│       └── master.html</br>
├── data</br>
│   ├── disaster_categories.csv</br>
│   ├── disaster_messages.csv</br>
│   ├── DisasterResponse.db</br>
│   └── process_data.py</br>
├── __init__.py</br>
├── license.txt</br>
├── models</br>
│   ├── __init__.py</br>
│   ├── test_0.pkl</br>
│   └── train_classifier.py</br>
├── README.md</br>
└── requirements.txt</br>

    
    - app - contains all app running assets.**templates** directory has html pages that the app serves.**run.py** has code to run the trained model
    - data - messages and categories csvs are used as trianing data for model generation. **process_data.py** performs ETL part of the pipeline. **DisasterResponse.db** is the product of the ETL pipeline
    - models - **train_classifier.py** Generates the trained model as pickle file. **test_0.pkl** is the trained model used for predictions.
    - requirements.txt - Python dependencied file
    
4. Results  
	There are two visualizarion components on the page
    1. Bar chart demonstrates the proportion of the genre of the the messages
    2. Textbox to input disaster messages and classify the message into a set of 36 predefined categories. The result is displayed in a new page with the categories returned by trained model highlighted. Highlighted categories also have respective probabilities displayed. This helps the user prioritize when there is a need. A side note on model training - model is trained by prioritizing *recall* metric while selecting best parameters during parameter optimization process. This process tends to increase the number of false positives but this is acceptable as a false positive is just a wrong category being activated and raising a false alarm. Is is assumed that disaster relief agencies are trained to handle false alarms.

5. ### License
    MIT license 
   
### Instructions to run the app:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
