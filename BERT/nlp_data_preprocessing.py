# NLP Disaster Prediction by Tweets - INITIAL DATA LOADING AND PREPROCESSING

import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import spacy
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from spacy.lang.en.examples import sentences 

warnings.filterwarnings('ignore')
nlp = spacy.load("en_core_web_sm")


class InitialDataLoader():
    def __init__(self, train_csv_dir, test_csv_dir):
        self.train_csv_dir = train_csv_dir
        self.test_csv_dir = test_csv_dir

    def data_preprocessing(self):
        print('Initial Data Loading and Preprocessing...')
        train_df = pd.read_csv(self.train_csv_dir)
        test_df = pd.read_csv(self.test_csv_dir)
        print('There are', len(train_df.index), 'training samples')
        print('There are', len(test_df.index), 'testing samples')

        train_text_samples = train_df.target.value_counts()
        sns.set(rc={'figure.figsize':(5,5)})

        sns.barplot(x = train_text_samples.index, y = train_text_samples)
        plt.gca().set_xlabel('Classes')
        plt.gca().set_ylabel('# of Samples')
        plt.savefig("./plots/class-distribution.png")
        print('There are', len(train_df[train_df['target'] == 0]['text']), 'samples are labeled as non-disaster')
        print('There are', len(train_df[train_df['target'] == 1]['text']), 'samples are labeled as disaster')
        #print('\nWe observe slight class imbalance but not critical')

        train_df['location'] = train_df['location'].fillna('None')
        train_df['keyword'] = train_df['keyword'].fillna('None')
        test_df['location'] = test_df['location'].fillna('None')
        test_df['keyword'] = test_df['keyword'].fillna('None')
        print('1. N/A filled.')

        train_df["text"] = train_df["text"].apply(lambda x: x.lower())
        test_df["text"] = test_df["text"].apply(lambda x: x.lower())
        print('2. Lowercased.')

        def spacy_punct(text):
            punct = []
            doc = nlp(text) #necessary to use SpaCy
            punct = [token.lemma_ for token in doc if token.is_punct]
            return punct

        train_df['punct'] = train_df['text'].apply(spacy_punct)
        train_df['punct'] = [' '.join(map(str, l)) for l in train_df['punct']]
        #train_df.head()

        punct_col = train_df['punct'].tolist()
        punct_list = []
        for sublist in punct_col:
            for item in sublist:
               punct_list.append(item)
        punct_freq = dict(Counter(punct_list))
        punct_freq = {i: j for i, j in sorted(punct_freq.items(), key=lambda item: item[1], reverse=True)}
        del punct_freq[' ']
        sns.set(rc={'figure.figsize':(8,6)})
        punct_keys = list(punct_freq.keys())
        punct_vals = list(punct_freq.values())
        sns.barplot(x = punct_keys, y = punct_vals)
        plt.savefig("./plots/punctuation-freqs.png")
        train_df = train_df.drop('punct', axis=1)

        def spacy_clean(text):
            preprocessed = []
            doc = nlp(text)
            preprocessed = [token.lemma_ for token in doc if not token.is_stop and not nlp.vocab[token.lemma_].is_stop and not token.is_punct and not token.is_digit and not token.like_url and not token.like_email and token.is_ascii]
            return preprocessed

        train_df['new_text'] = train_df['text'].apply(spacy_clean)
        test_df['new_text'] = test_df['text'].apply(spacy_clean)
        train_df['new_text'] = [' '.join(map(str, l)) for l in train_df['new_text']]
        test_df['new_text'] = [' '.join(map(str, l)) for l in test_df['new_text']]
        print('3. Tokenized, Lemmanized, Stop Words Removed.')

        train_df['new_text'] = train_df["new_text"].str.replace('[^\w\s]', "").str.replace('[0-9]', "").str.replace(' [a-z] ', "").str.replace('-', "").str.replace('_', "").str.replace(' amp ', "")
        test_df['new_text'] = test_df["new_text"].str.replace('[^\w\s]', "").str.replace('[0-9]', "").str.replace(' [a-z] ', "").str.replace('-', "").str.replace('_', "").str.replace(' amp ', "")
        print('4. Special Characters and Numbers Removed.')
        #train_df.head()
        #test_df.head()

        word_cloud_before1 = '  '.join(list(train_df[train_df['target'] == 1]['text']))
        word_cloud_before1 = WordCloud(background_color='white', width = 500, height = 400).generate(word_cloud_before1)
        word_cloud_after1 = '  '.join(list(train_df[train_df['target'] == 1]['new_text']))
        word_cloud_after1 = WordCloud(background_color='white', width = 500, height = 400).generate(word_cloud_after1)

        fig, ax = plt.subplots(1, 2, figsize=(16, 14))
        ax[0].imshow(word_cloud_before1)
        ax[1].imshow(word_cloud_after1)

        ax[0].title.set_text('Before Text Preprocessing\n')
        ax[1].title.set_text('After Text Preprocessing\n')
        ax[0].figure.savefig("./plots/word-cloud-target-1.png")
        #plt.show()

        word_cloud_before0 = '  '.join(list(train_df[train_df['target'] == 0]['text']))
        word_cloud_before0 = WordCloud(background_color='white', width = 400, height = 300).generate(word_cloud_before0)
        word_cloud_after0 = '  '.join(list(train_df[train_df['target'] == 0]['new_text']))
        word_cloud_after0 = WordCloud(background_color='white', width = 400, height = 300).generate(word_cloud_after0)

        fig, ax = plt.subplots(1, 2, figsize=(16, 14))
        ax[0].imshow(word_cloud_before0)
        ax[1].imshow(word_cloud_after0)

        ax[0].title.set_text('Before Text Preprocessing\n')
        ax[1].title.set_text('After Text Preprocessing\n')
        ax[0].figure.savefig("./plots/word-cloud-target-0.png")
        #plt.show()

        dis_train_tweets_freq = train_df[train_df['target'] == 0]['new_text'].value_counts()
        #print(dis_train_tweets_freq)

        dis_train_tweets_freq = train_df[train_df['target'] == 1]['new_text'].value_counts()
        #print(dis_train_tweets_freq)

        train_df_new = train_df.drop_duplicates(subset='new_text', keep="first")
        #train_df_new.head()
        print('5. Duplicates Dropped.')

        dis_train_words_freq1 = train_df_new[train_df_new['target'] == 1]['new_text'].str.split(expand=True).stack().value_counts()
        dis_train_words_freq0 = train_df_new[train_df_new['target'] == 0]['new_text'].str.split(expand=True).stack().value_counts()

        sns.set(rc={'figure.figsize':(12,9)})
        fig, ax = plt.subplots(1, 2)

        sns.barplot(ax = ax[0], x = dis_train_words_freq1[:30], y = dis_train_words_freq1.index[:30])
        ax[0].set_title('Target = 1')
        ax[0].set_xlabel('Frequencies')
        ax[0].set_ylabel('Words')
        sns.barplot(ax = ax[1], x = dis_train_words_freq0[:30], y = dis_train_words_freq0.index[:30])
        ax[1].set_title('Target = 0')
        ax[1].set_xlabel('Frequencies')
        ax[0].figure.savefig("./plots/words-freqs.png")
        print('Data Has Been Fully Preprocessed!')
        return train_df_new, test_df

#initial_data_loader = InitialDataLoader('C:\\Users\\a_lite13\\Dropbox\\NLP-Disaster-Tweets\\train.csv', 'C:\\Users\\a_lite13\\Dropbox\\NLP-Disaster-Tweets\\test.csv')
#initial_data = initial_data_loader.data_preprocessing()
#print(initial_data[0]['new_text'])

