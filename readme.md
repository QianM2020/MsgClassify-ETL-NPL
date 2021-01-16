
#Project Overview

  In this project, I analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages.

#Goal

  A machine learning pipeline to categorize the disaster events are created in
  this project, so that it can send the messages to an appropriate disaster
  relief agency/category.

#Environment

  The virtual workspace IDE and data set are offered by Udacity.


#Dataset

  The data set containing real messages that were sent during disaster events.
  The raw data: 'disaster_categories.csv' and 'disaster_messages.csv' can be accessed in folder 'data'.

  I used the ETL pipeline to clean the data. The cleaned data is saved as 'DisasterResponse.db' in folder 'data'.
  
  * Insights from the dataset
  	This dataset is imbalanced. Some labels like water,child-alone,tools, shops, firs, cold have very few data, while some like 'related', 'aid-related','Direct Report' have more examples.
  	To some extend, this imbalance affects the model training. I noted that the latter 3 categories tend to be the easy-predicted results with my model.
  	Very probably that more weights are put on these 3-4 categories when compute precision or recall for the various categories during the training process.
  	Thus these categories turn to be match most cases, in other words, they are more possible to be predicted.
  	
  	One solution maybe to adjust the weights of different categories to compute their precisions or recalls.
 
#model

  The trained model is stored in a pickle file: 'classifier.pkl' under folder 'models'.

#Components

  *Flask Web App:
    where an emergency worker can input a new message and get classification
    results in several categories.
    The web app will also display visualizations of the message genres and categoris distribution.

    You can find the html files of the 2 web pages in folder 'app':
      'master.html' and 'go.html'
    The app was run on Udacity's virtual server. The host and port are specified as 'host='0.0.0.0', port=3001'.

    You can find find 'run.py' in folder 'app'. Run the following command in the app's directory to run your web app.`python run.py` and Go to the specified host and port, you are supposed to access the webpage.


  *ETL pipeline:
    you can find its code in 'process_data.py' under folder 'data'. Its functions include:
      Loads the messages and categories datasets
      Merges the two datasets
      Cleans the data
      Stores it in a SQLite database

  *ML Pipeline:
    you can find its code in t'rain_classifier.py' under folder 'model'. Its functions include:
      Loads data from the SQLite database
      Splits the dataset into training and test sets
      Builds a text processing and machine learning pipeline
      Trains and tunes a model using GridSearchCV
      Outputs results on the test set
      Exports the final model as a pickle file

      In addtion to the ML pipeline, you can find 2 extra modules:
        'ChecktagExtractor.py' can get count the token's frequency in a message according to their word category. This can help model to approach more features of the data.
        'Tokennize.py' can tokenize and lemmatize the message contents.
