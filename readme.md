
# Project Overview

   In this project, we analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages.

## Background
   Every day, disaster organizations get millions and millions of disaster communications either direct or via social media. Often 1 in every 1000 these messages might be relevant to the disaster response profesionals. Typical different organnizations will respond to different type of msgs and take care of different parts of disaster. The organization expect the least capacity to filter and pull out the msgs which are the most important. The basic idea of this project is to use 'Figure 8' technology for the disasters from which these msgs are taken, then combined these datasets and relabeled them to correponding categories so that they are consistent labels across the different disasters, and to build supervised ML models.


## Dataset 

  The data set containing real messages that were sent during disaster events.
#### Raw data
  * 'disaster_categories.csv'   
  
  * 'disaster_messages.csv' 
  
#### Cleaned data
  * 'DisasterResponse.db' (cleaned by ETL pipeline) 

## Components

  * ETL Pipeline ('process_data.py')
      * Loads the messages and categories datasets
      * Merges the two datasets
      * Cleans the data
      * Stores it in a SQLite database

  * ML Pipeline ('train_classifier.py')
      * Loads data from the SQLite database
      * Splits the dataset into training and test sets
      * Builds a text processing and machine learning pipeline
      * Trains and tunes a model using GridSearchCV
      * Outputs results on the test set
      * Exports the final model as a pickle file (classifier.pkl)
	
 * Flask Web App 
    
    where an emergency worker can input a new message and get classification results in several categories.
    The web app will also display visualizations of the message genres and categoris distribution.
      * master.html
      * go.html

## Run Web App

   * Type in `python run.py` in the Terminal Window.
   * Go to https://view6914b2f4-3001.udacity-student-workspaces.com/.
   Note: The app run on Udacity's virtual server. The host and port are specified as 'host='0.0.0.0', port=3001'.
   
    
## File Structure

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
