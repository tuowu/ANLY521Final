import pandas as pd
import string
import numpy as np
import re
import en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS
from imblearn.over_sampling import RandomOverSampler, SMOTE
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from sklearn.metrics import classification_report
import argparse
from keras.utils.np_utils import to_categorical


def CNN(train_X, test_X, train_y, test_y):
    model = Sequential()
    model.add(Embedding(20000, 100, input_length=50))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, 2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_X, train_y,
              batch_size=128,
              epochs=20,
              verbose=1,
              validation_data=(test_X, test_y),
              shuffle=True)
    preds = model.predict(test_X)
    print(classification_report(np.argmax(test_y, axis=1), np.argmax(preds, axis=1)))

# preprocessing method 1
def spacy_token(text):
    text = re.sub(citation_pattern, 'CIT', text)
    text = re.sub('\-', ' ', text)
    text = re.sub('\d', '', text)
    mytokens = nlp(text)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in stopwords and word not in punctuations]
    return ' '.join(mytokens)


# preprocessing method 2
def new_spacy_token(text):
    text = re.sub(citation_pattern, 'cit', text)
    text = re.sub('\-', ' ', text)
    my_token = nlp(text)
    overall_list = []
    for token in my_token:
        if token.lemma_ == 'cit':
            overall_list.append(token.lemma_)
        if token.pos_ == 'VERB' and token.lemma_ != 'cit':
            overall_list.append(token.lemma_)
        if token.pos_ == 'ADV' and token.lemma_ != 'cit':
            overall_list.append(token.lemma_)
        if token.pos_ == 'ADJ' and token.lemma_ != 'cit':
            overall_list.append(token.lemma_)
        if token.pos_ == 'NOUN' and token.lemma_ != 'cit':
            overall_list.append(token.lemma_)
    return ' '.join([w for w in overall_list if w not in punctuations])


def main(arg):
    print("start loading data")
    df_train = pd.read_csv(arg.train_file)
    df_test = pd.read_csv(arg.val_file)
    df_train_spacy = pd.read_csv('train_spacy_simulated.csv')
    df_train_newSpacy = pd.read_csv('train_newSpacy_simulated.csv')
    df_train_spacy.columns = ['new_text', 'labels']
    df_train_newSpacy.columns = ['new_text', 'labels']
    df_train_spacy_n = df_train_spacy[df_train_spacy.labels == 0]
    df_train_spacy_p = df_train_spacy[df_train_spacy.labels == 1]
    df_train_newSpacy_n = df_train_newSpacy[df_train_newSpacy.labels == 0]
    df_train_newSpacy_p = df_train_newSpacy[df_train_newSpacy.labels == 1]
    print("finished loading data")

    global nlp, stopwords, punctuations, citation_pattern
    citation_pattern = r'\((.*?)\)'
    nlp = en_core_web_sm.load()
    stopwords = list(STOP_WORDS)
    punctuations = string.punctuation

    print("start proprocessing")
    if arg.augment:
        if arg.preprocessing_method == 'p1':
            df_train['new_text'] = df_train['text'].apply(spacy_token)
            df_test['new_text'] = df_test['text'].apply(spacy_token)
            df_train = df_train[['new_text', 'labels']]
            df_test = df_test[['new_text', 'labels']]
            if arg.augment_class == 'Negative':
                temp_df = df_train_spacy_n.sample(n=int(arg.augment_rate * len(df_train_spacy_n)))
                df_train = df_train.append(temp_df, ignore_index=True)
            elif arg.augment_class == 'Positive':
                temp_df = df_train_spacy_p.sample(n=int(arg.augment_rate * len(df_train_spacy_p)))
                df_train = df_train.append(temp_df, ignore_index=True)
            elif arg.augment_class == 'All':
                temp_df = df_train_spacy_n.sample(n=int(arg.augment_rate * len(df_train_spacy_n)))
                df_train = df_train.append(temp_df, ignore_index=True)
                temp_df = df_train_spacy_p.sample(n=int(arg.augment_rate * len(df_train_spacy_p)))
                df_train = df_train.append(temp_df, ignore_index=True)

        elif arg.preprocessing_method == 'p2':
            df_train['new_text'] = df_train['text'].apply(new_spacy_token)
            df_test['new_text'] = df_test['text'].apply(new_spacy_token)
            df_train = df_train[['new_text', 'labels']]
            if arg.augment_class == 'Negative':
                temp_df = df_train_newSpacy_n.sample(n=int(arg.augment_rate * len(df_train_newSpacy_n)))
                df_train = df_train.append(temp_df, ignore_index=True)
            elif arg.augment_class == 'Positive':
                temp_df = df_train_newSpacy_p.sample(n=int(arg.augment_rate * len(df_train_newSpacy_p)))
                df_train = df_train.append(temp_df, ignore_index=True)
            elif arg.augment_class == 'All':
                temp_df = df_train_newSpacy_n.sample(n=int(arg.augment_rate * len(df_train_newSpacy_n)))
                df_train = df_train.append(temp_df, ignore_index=True)
                temp_df = df_train_newSpacy_p.sample(n=int(arg.augment_rate * len(df_train_newSpacy_p)))
                df_train = df_train.append(temp_df, ignore_index=True)
    else:
        if arg.preprocessing_method == 'p1':
            df_train['new_text'] = df_train['text'].apply(spacy_token)
            df_test['new_text'] = df_test['text'].apply(spacy_token)
        elif arg.preprocessing_method == 'p2':
            df_train['new_text'] = df_train['text'].apply(new_spacy_token)
            df_test['new_text'] = df_test['text'].apply(new_spacy_token)

    train_X, test_X, train_y, test_y = list(df_train['new_text']), list(
        df_test['new_text']), df_train.labels.values, df_test.labels.values
    print("finished preprocessing")
    print("Start Vectorizering")
    text_corpus = train_X.copy()
    text_corpus.extend(test_X)
    tk = Tokenizer(lower=True, filters='')
    tk.fit_on_texts(text_corpus)
    max_len = 50
    train_tokenized = tk.texts_to_sequences(train_X)
    test_tokenized = tk.texts_to_sequences(test_X)
    train_X = pad_sequences(train_tokenized, maxlen=max_len)
    test_X = pad_sequences(test_tokenized, maxlen=max_len)
    train_y = to_categorical(train_y, num_classes= 3)
    test_y = to_categorical(test_y, num_classes = 3)
    print("finished Vectorizering")
    print(train_X.shape)
    print(train_y.shape)
    print(test_X.shape)
    print(test_y.shape)

    if arg.oversample:
        print("Start Oversampling")
        ros = RandomOverSampler()
        train_X, train_y = ros.fit_sample(train_X, train_y)
        print("Oversample finished")

    print("Start CNN")
    CNN(train_X, test_X, train_y, test_y)
    print("Finished CNN")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str,
                        default="train_original.csv",
                        help="training_file")
    parser.add_argument("--val_file", type=str,
                        default="test_original.csv", )
    parser.add_argument("--preprocessing_method", type=str,
                        default='p1')
    parser.add_argument("--oversample", type=bool,
                        default=False)
    parser.add_argument("--augment", type=bool,
                        default=False)
    parser.add_argument("--augment_class", type=str,
                        default='Negative')
    parser.add_argument("--augment_rate", type=float,
                        default=.3)

    args = parser.parse_args()

    main(args)
