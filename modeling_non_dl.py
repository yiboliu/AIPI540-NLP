import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

from data_processing import preprocess, select_words, select_words_for_each_sent


def build_features(data, ngram_range):
    vec = TfidfVectorizer(ngram_range=ngram_range)
    transformed = vec.fit_transform(data)
    return transformed, vec


def train_and_test(X, y):
    num_folds = 5
    model = LogisticRegression(solver='saga', max_iter=5000)
    for i in range(num_folds):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        print(accuracy_score(y_val, preds))
    return model


def get_stats_df(data):
    vals = []
    for item in data['selected_text']:
        vals.append(len(item.split()))
    counts = np.array(vals)
    return np.mean(counts), counts.max()


def launch_training_nondl(data, model_path, vec_path):
    avg, max_len = get_stats_df(df)

    X_train, vec = build_features(data['selected_text'], (1, max_len))
    y_train = data['sentiment']
    model = train_and_test(X_train, y_train)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vec_path, 'wb') as f:
        pickle.dump(vec, f)


def serve_model_non_dl(model_path, vec_path, sentence):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vec_path, 'rb') as f:
        vec = pickle.load(f)
    preprocessed = preprocess(sentence)
    words = select_words_for_each_sent(preprocessed, 2)
    trans = vec.transform([words])
    features = trans.toarray()

    return model.predict(features)[0]


if __name__ == "__main__":
    df = select_words('train.csv')
    non_dl_model_path = 'models/model-lr.pkl'
    vec_path = 'models/vec.pkl'
    launch_training_nondl(df, non_dl_model_path, vec_path)
