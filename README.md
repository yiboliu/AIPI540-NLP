# AIPI540-NLP
This is the NLP project for course AIPI 540 at Duke University

## Overview
This project mainly deals with identifying tweets that are positive, negative or neutral in terms of sentiments. Tweets with different sentiments can affect reader's emotions in different ways, so it is important to identify the sentiments and use it as a factor of whether and to whom to present the tweets. This is to build a healthier community. 

## Uniqueness
A little bit different from the description of the project on [Kaggle](https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview), my work here contains 2 parts: 
1. preprocess the words: to remove stopwords, lemmatize and extract the list of words that most effectively embody the sentiments;
2. determine whether the extracted list of words indicate positive/negative/neutral sentiments; (In the original Kaggle project this is given)

The biggest difference here is that I disregarded the column of selected words from the original data but use my own data. 

## Models
The first model is the Deep Learning model, which is trained based on a embedding layer and a fully connected linear layer. One thing to notice here is that I tried to add a pre trained GloVe layer to enhance the complexity, in order to improve the model performance, but it turned out that accuracy was lowered rather than improved. So I removed that layer and kept the simple form.

The second model is the non-DL model, which utilizes TF-IDF to create features and feed a logistic regression model to predict sentiments. 

## Setup

To set up the project, just install all the dependencies in the requirements.txt, by running ``pip install -r requirements.txt`` and run ``python setup.py`` to decompress the data files

To train the deep learning model, run ``python modeling_dl.py``. This will train the model and save the related artifacts in `models/` directory for later use.

To train the non deep learning model, run ``python modeling_non_dl.py``. This will train the model and save the related artifacts in `models/` directory for later use.

## Demo

To have a try with the demo, run ``streamlit run main.py --server.port=8080 --server.address=0.0.0.0``. (Please bear for a few seconds for it to fully start) You will see a prompt for entering a sentence and you will see the results from both DL and non DL models.

## Results Comparison

Unfortunately, the accuracies from the both models were around 0.57 and couldn't improve. The deep learning model with a GloVe layer was even worse, 0.52. 
I tried a bunch of different combinations of hyperparameters, and got the optimal results were around 0.57. 
I tried a learning rate of 0.001, but the accuracy was stuck at 0.34. Then I changed the learning rate to be 1, 5, and 10, also adjusting the batch size, gemma and step sizes, the resulting accuracy barely increased over 0.57. 
The reason I think that this result was bad is due to the size of the dataset. The vocab size was 16903, which could not provide sufficient data to train a model. 