# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Table of Contents
1. ### Installation
    This code is run on Python 3.6.x. Please install all the python dependencies using **requirements.txt**
2. ### Project Motivation
	 Analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. Using real messages that were sent during disaster events, goal is to create a machine learning pipeline to categorize these events and categorize them into 36 categories. Categories extracted from input data. An an emergency worker can input a new message and get classification results in several categories.
    
3. ### Files and their description
├── app
│   ├── run.py
│   └── templates
│       ├── go.html
│       └── master.html
├── data
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   └── process_data.py
├── DisasterResponse.db
├── license.txt
├── models
│   └── train_classifier.py
├── README.md
├── test_0.pkl
├── test.pkl
	
    * app - contains all app running assets.**templates** directory has html pages that the app serves.**run.py** has code to run the trained model
    * data - messages and categories csvs are used as trianing data for model generation. **process_data.py** performs ETL part of the pipeline
    * models - **train_classifier.py** Generates the trained model as pickle file
    
    
4. ### Results
   - 
   -
   

5. ### License
   
