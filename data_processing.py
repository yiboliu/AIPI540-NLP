import csv
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import networkx as nx
import pandas as pd


def preprocess(sentence):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()  # lemmatize each word
    sentence = sentence.lower()  # to lower cases
    tokenizer = RegexpTokenizer(r'\w+')  # tokenize and remove punctuations
    words = tokenizer.tokenize(sentence)
    results = []
    for word in words:
        if word in stop_words:
            continue
        lemmatized = lemmatizer.lemmatize(word)
        results.append(lemmatized)

    return results


def select_words_for_each_sent(preprocessed, window_size):
    G = nx.Graph()
    for word in preprocessed:
        G.add_node(word)

    for i in range(len(preprocessed)):
        for distance in range(1, window_size):
            if i + distance >= len(preprocessed):
                break
            left = preprocessed[i]
            right = preprocessed[i+distance]
            if G.has_edge(left, right):
                G[left][right]['weight'] += 1
            else:
                G.add_edge(preprocessed[i], preprocessed[i+distance], weight=1)
    scores = nx.pagerank(G)
    selected = [k for k in scores if scores[k] >= float(1.0 / len(scores))]
    return " ".join(selected)


def select_words(file_name):
    df = pd.DataFrame(columns=['textID', 'selected_text', 'sentiment'])
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cnt = 0
        for row in reader:
            preprocessed = preprocess(row['text'])
            selected_text = select_words_for_each_sent(preprocessed, 2)
            if not selected_text:
                continue
            cnt += 1
            df = pd.concat([df, pd.DataFrame({
                'textID': [row['textID']],
                'selected_text': [selected_text],
                'sentiment': [row['sentiment']]
            })])
        df.index = pd.RangeIndex(start=0, stop=cnt, step=1)
    return df
