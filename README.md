# AIPI540-NLP
This is the NLP project for course AIPI 540 at Duke University

## Overview
This project mainly deals with identifying tweets that are positive, negative or neutral in terms of sentiments. Tweets with different sentiments can affect reader's emotions in different ways, so it is important to identify the sentiments and use it as a factor of whether and to whom to present the tweets. This is to build a healthier community. 

## Uniqueness
A little bit different from the description of the project on [Kaggle](https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview), my work here contains 3 parts: 
1. extract the list of words that most effectively embody the sentiments;
2. determine whether the extracted list of words indicate positive/negative/neutral sentiments; (In the original Kaggle project this is given)
3. generate a paraphrased sentence to the original sentence. 


## Models
The first model is the Deep Learning model, which performs the task of extractive summarization. 

The second model is the non-DL model, which utilizes Bag Of Words to predict sentiments. 