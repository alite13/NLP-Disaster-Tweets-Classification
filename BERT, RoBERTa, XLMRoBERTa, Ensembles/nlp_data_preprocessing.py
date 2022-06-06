# NLP Disaster Prediction by Tweets - INITIAL DATA LOADING AND PREPROCESSING

import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import spacy
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from skimage import io

warnings.filterwarnings('ignore')
nlp = spacy.load("en_core_web_sm")


class InitialDataLoader():
    def __init__(self, train_csv_dir, test_csv_dir):
        self.train_csv_dir = train_csv_dir
        self.test_csv_dir = test_csv_dir

    def train_data_preprocessing(self):
        print('\n======== Initial Data Loading and Preprocessing ========\n')

        """ .csv datasets load """

        train_df = pd.read_csv(self.train_csv_dir)
        test_df = pd.read_csv(self.test_csv_dir)
        print('There are', len(train_df.index), 'training samples and', len(test_df.index), 'testing samples loaded from files.')

        """ Find and remove duplicate tweets """

        train_df = train_df.drop_duplicates(subset='text', keep="first")
        print('<<< Duplicate tweets removed >>>')
        #print('There are', len(train_df.index), 'training samples now.')

        """ Display the distribution of samples """

        train_text_samples = train_df.target.value_counts()
        sns.set(rc={'figure.figsize':(20,15)})
        fig, ax = plt.subplots(2, 2)
        colors = ['salmon' if (x < max(train_text_samples)) else 'yellowgreen' for x in train_text_samples]
        sns.barplot(ax = ax[0][0], x = train_text_samples.index, y = train_text_samples, palette = colors)      

        ax[0][0].set_xlabel('Classes')
        ax[0][0].set_ylabel('# of Samples')
        ax[0][0].set_title('Distribution of Training Tweets')
        #print('There are', len(train_df[train_df['target'] == 0]['text']), 'samples are labeled as non-disaster.')
        #print('There are', len(train_df[train_df['target'] == 1]['text']), 'samples are labeled as disaster.')

        """ Display the percentage of NAs and replace them """

        percent_missing = train_df.isnull().sum() * 100 / len(train_df)
        missing_vals = pd.DataFrame({'col_name': train_df.columns,
                                 'percent': percent_missing})
        cols = list(missing_vals['col_name'])
        percent = list(missing_vals['percent'])
        sns.barplot(ax = ax[0][1], x = cols, y = percent)

        ax[0][1].set_ylabel('% of Missing Values')
        ax[0][1].set_xlabel('Column Name')
        ax[0][1].set_title('% of Missing Train Values Among Columns')

        train_df['location'] = train_df['location'].fillna('None')
        train_df['keyword'] = train_df['keyword'].fillna('None')
        print('<<< N/A values filled >>>')

        """ Lowercasing """

        train_df['text'] = train_df['text'].apply(lambda x: x.lower())
        train_df['location'] = train_df['location'].apply(lambda x: x.lower())
        print('<<< Lowercased >>>')

        """ Visualize all punctuations """

        def spacy_punct(text):
            punct = []
            doc = nlp(text) #necessary to use SpaCy
            punct = [token.lemma_ for token in doc if token.is_punct]
            return punct
        train_df['punct'] = train_df['text'].apply(spacy_punct)
        train_df['punct'] = [' '.join(map(str, l)) for l in train_df['punct']]
        
        punct_col = train_df['punct'].tolist()
        punct_list = []
        for sublist in punct_col:
            for item in sublist:
               punct_list.append(item)
        punct_freq = dict(Counter(punct_list))
        punct_freq = {i: j for i, j in sorted(punct_freq.items(), key=lambda item: item[1], reverse=True)}
        del punct_freq[' ']

        punct_keys = list(punct_freq.keys())
        punct_vals = list(punct_freq.values())
        sns.barplot(ax = ax[1][0], x = punct_keys, y = punct_vals)
        ax[1][0].set_title('Common Punctuations')
        ax[1][0].set_ylabel('Frequency')
        ax[1][0].set_xlabel('Punctuations')
        train_df = train_df.drop('punct', axis=1)

        """ Visualize top locations """

        def spacy_location(text):
            preprocessed = []
            doc = nlp(text)
            preprocessed = [token.lemma_ for token in doc if not token.is_punct and not token.is_digit and not token.like_url and not token.like_email and token.is_ascii]
            return preprocessed
        train_df['location_names'] = train_df['location'].apply(spacy_location)
        train_df['location_names'] = [' '.join(map(str, l)) for l in train_df['location_names']]
        train_df['location_names'] = train_df['location_names'].str.replace('[^\w\s]', "").str.replace('[0-9]', "").str.replace(' [a-z] ', "").str.replace(' [a-z][a-z]', "").str.replace('-', "").str.replace('_', "").str.replace('@', "")

        location_col = train_df['location_names'].tolist()
        location_freq = dict(Counter(location_col))
        location_freq = {i: j for i, j in sorted(location_freq.items(), key=lambda item: item[1], reverse=True)}
        del location_freq[' ']
        del location_freq['']
        del location_freq['none']
        location_keys = list(location_freq.keys())
        location_vals = list(location_freq.values())
        sns.barplot(ax = ax[1][1], x = location_keys[:20], y = location_vals[:20])
        ax[1][1].set_title('Top Locations')
        ax[1][1].set_ylabel('Frequency')
        ax[1][1].set_xlabel('Location Name')
        ax[1][1].set_xticklabels(ax[1][1].get_xticklabels(), rotation = 45)
        train_df = train_df.drop('location_names', axis=1)
        ax[0][0].figure.savefig("./plots/eda.png")
        

        """ Basic Cleaning - tokenization, URLs, digits, punctuations, non-ascii, emails removal """

        def spacy_clean(text):
            preprocessed = []
            doc = nlp(text)
            preprocessed = [token.lemma_ for token in doc if not token.is_punct and not token.is_digit and not token.like_url and not token.like_email and token.is_ascii]
            return preprocessed

        train_df['new_text'] = train_df['text'].apply(spacy_clean)
        train_df['new_text'] = [' '.join(map(str, l)) for l in train_df['new_text']]
        print('<<< Tokenized, URLs, digits, punctuations, non-ascii, emails removed >>>')
        train_df['new_text'] = train_df["new_text"].str.replace('[^\w\s]', "").str.replace('[0-9]', "").str.replace(' [a-z] ', "").str.replace('-', "").str.replace('_', "").str.replace('&amp', "").str.replace('@', "")
        print('<<< Special characters and numbers removed >>>')

        """ Advanced Cleaning - tokenization, lemmanization, stop words, URLs, digits, punctuations, non-ascii, emails removal """

        #def spacy_clean(text):
            #preprocessed = []
            #doc = nlp(text)
            #preprocessed = [token.lemma_ for token in doc if not token.is_stop and not nlp.vocab[token.lemma_].is_stop and not token.is_punct and not token.is_digit and not token.like_url and not token.like_email and token.is_ascii]
            #return preprocessed

        #train_df['new_text'] = train_df['text'].apply(spacy_clean)
        #train_df['new_text'] = [' '.join(map(str, l)) for l in train_df['new_text']]
        #print('<<< Tokenized, lemmanized, stop words, URLs, digits, punctuations, non-ascii, emails removed >>>')
        #train_df['new_text'] = train_df["new_text"].str.replace('[^\w\s]', "").str.replace('[0-9]', "").str.replace(' [a-z] ', "").str.replace('-', "").str.replace('_', "").str.replace(' amp ', "").str.replace('@', "")
        #print('<<< Special characters and numbers removed >>>')
        
        """ Advanced Cleaning - Remove Non-English words """

        #nltk_words = set(nltk.corpus.words.words())
        #preprocessed = []
        #for i in train_df['new_text']:
            #doc = nltk.wordpunct_tokenize(i)
            #preprocessed.append(" ".join(w for w in doc if w.lower() in nltk_words or not w.isalpha()))
            #train_df['new_text'] = preprocessed
        #train_df.head()

        """ Wordcloud for disaster tweets """

        mask = io.imread('https://raw.githubusercontent.com/rasbt/datacollect/master/dataviz/twitter_cloud/twitter_mask.png')

        word_cloud_before1 = '  '.join(list(train_df[train_df['target'] == 1]['text']))
        word_cloud_before1 = WordCloud(background_color='white', width = 400, height = 300, colormap='Set1', mask = mask).generate(word_cloud_before1)
        word_cloud_after1 = '  '.join(list(train_df[train_df['target'] == 1]['new_text']))
        word_cloud_after1 = WordCloud(background_color='white', width = 400, height = 300, colormap='Set1', mask = mask).generate(word_cloud_after1)

        fig, ax = plt.subplots(1, 2, figsize=(16, 11))
        ax[0].imshow(word_cloud_before1, interpolation="bilinear")
        ax[1].imshow(word_cloud_after1, interpolation="bilinear")
        ax[0].title.set_text('Before Text Preprocessing\n')
        ax[1].title.set_text('After Text Preprocessing\n')
        ax[0].axis('off')
        ax[1].axis('off')
        ax[0].figure.savefig('./plots/word-cloud-target-1.png')
        #plt.show()

        """Wordcloud for non-disaster tweets"""

        word_cloud_before0 = '  '.join(list(train_df[train_df['target'] == 0]['text']))
        word_cloud_before0 = WordCloud(background_color='white', width = 400, height = 300, mask = mask).generate(word_cloud_before0)
        word_cloud_after0 = '  '.join(list(train_df[train_df['target'] == 0]['new_text']))
        word_cloud_after0 = WordCloud(background_color='white', width = 400, height = 300, mask = mask).generate(word_cloud_after0)

        fig, ax = plt.subplots(1, 2, figsize=(16, 11))
        ax[0].imshow(word_cloud_before0, interpolation="bilinear")
        ax[1].imshow(word_cloud_after0, interpolation="bilinear")
        ax[0].title.set_text('Before Text Preprocessing\n')
        ax[1].title.set_text('After Text Preprocessing\n')
        ax[0].axis('off')
        ax[1].axis('off')
        ax[0].figure.savefig('./plots/word-cloud-target-0.png')
        #plt.show()

        """Find and delete duplicate tweets"""

        #train_tweets_freq0 = train_df[train_df['target'] == 0]['new_text'].value_counts()
        #train_tweets_freq0.head()

        #train_tweets_freq1 = train_df[train_df['target'] == 1]['new_text'].value_counts()
        #train_tweets_freq1.head()

        train_df_new = train_df.drop_duplicates(subset = 'new_text', keep = 'first')

        print('<<< Duplicate tweets removed >>>')

        """ Words frequency """

        train_words_freq1 = train_df_new[train_df_new['target'] == 1]['new_text'].str.split(expand = True).stack().value_counts()
        train_words_freq0 = train_df_new[train_df_new['target'] == 0]['new_text'].str.split(expand = True).stack().value_counts()

        sns.set(rc = {'figure.figsize':(12,9)})
        fig, ax = plt.subplots(1, 2)

        sns.barplot(ax = ax[0], x = train_words_freq1[:30], y = train_words_freq1.index[:30])
        ax[0].set_title('Target = 1')
        ax[0].set_xlabel('Frequency')
        ax[0].set_ylabel('Words')

        sns.barplot(ax = ax[1], x = train_words_freq0[:30], y = train_words_freq0.index[:30])
        ax[1].set_title('Target = 0')
        ax[1].set_xlabel('Frequency')
        ax[1].set_ylabel('Words')
        ax[0].figure.savefig("./plots/words-freqs.png")
        #plt.show()

        """N-grams analysis"""

        def bigram(corpus, n = None):
            vectorizer = CountVectorizer(ngram_range = (2, 2)).fit(corpus)
            bag_of_words = vectorizer.transform(corpus)
            sum_words = bag_of_words.sum(axis = 0) 
            words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
            words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
            return words_freq[:n]

        def threegram(corpus, n = None):
            vectorizer = CountVectorizer(ngram_range = (3, 3)).fit(corpus)
            bag_of_words = vectorizer.transform(corpus)
            sum_words = bag_of_words.sum(axis = 0) 
            words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
            words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
            return words_freq[:n]

        fig, ax = plt.subplots(1, 2)
        sns.set(rc = {'figure.figsize':(16, 7)})
        ax[0].set_title('2-grams')
        ax[0].set_xlabel('Frequency')
        ax[0].set_ylabel('')
        top_bigrams = bigram(train_df_new['new_text'])[:20]
        x, y = map(list,zip(*top_bigrams))
        sns.barplot(ax = ax[0], x = y, y = x)

        ax[1].set_xlabel('Frequency')
        ax[1].set_ylabel('')
        ax[1].set_title('3-grams')
        top_threegrams = threegram(train_df_new['new_text'])[:20]
        x1, y1 = map(list,zip(*top_threegrams))
        sns.barplot(ax = ax[1], x = y1, y = x1)
        plt.savefig("./plots/bigrams-threegrams.png")

        """ Distribution of characters in disaster tweets (target = 1) """

        sns.set(rc={'figure.figsize':(12,5)})
        fig, ax = plt.subplots(1, 2)

        text_len1 = train_df[train_df['target'] == 1]['text'].str.len()
        ax[0].hist(text_len1, color = "salmon")
        ax[0].set_title('Before preprocessing')
        ax[0].set_xlabel('Number of Characters')
        ax[0].set_ylabel('Frequency')

        text_len2 = train_df_new[train_df_new['target'] == 1]['new_text'].str.len()
        ax[1].hist(text_len2, color = "salmon")
        ax[1].set_title('After preprocessing')
        ax[1].set_xlabel('Number of Characters')
        ax[1].set_ylabel('Frequency')
        fig.suptitle('Distribution of Characters in Disaster Tweets (target = 1)')
        ax[0].figure.savefig("./plots/distribution-characters1.png")
        #plt.show()

        """ Distribution of characters in disaster tweets (target = 0) """

        sns.set(rc={'figure.figsize':(12,5)})
        fig, ax = plt.subplots(1, 2)

        text_len1 = train_df[train_df['target'] == 0]['text'].str.len()
        ax[0].hist(text_len1, color = "yellowgreen")
        ax[0].set_title('Before preprocessing')
        ax[0].set_xlabel('Number of Characters')
        ax[0].set_ylabel('Frequency')

        text_len2 = train_df_new[train_df_new['target'] == 0]['new_text'].str.len()
        ax[1].hist(text_len2, color = "yellowgreen")
        ax[1].set_title('After preprocessing')
        ax[1].set_xlabel('Number of Characters')
        ax[1].set_ylabel('Frequency')
        fig.suptitle('Distribution of Characters in Non-Disaster Tweets (target = 0)')
        ax[0].figure.savefig("./plots/distribution-characters0.png")
        #plt.show()

        """ Distribution of words in disaster tweets (target = 1) """

        sns.set(rc={'figure.figsize':(12,5)})
        fig, ax = plt.subplots(1, 2)

        text_len1 = train_df[train_df['target'] == 1]['text'].str.split().map(lambda x: len(x))
        ax[0].hist(text_len1, color = "salmon")
        ax[0].set_title('Before preprocessing')
        ax[0].set_xlabel('Number of Words')
        ax[0].set_ylabel('Frequency')

        text_len2 = train_df_new[train_df_new['target'] == 1]['new_text'].str.split().map(lambda x: len(x))
        ax[1].hist(text_len2, color = "salmon")
        ax[1].set_title('After preprocessing')
        ax[1].set_xlabel('Number of Words')
        ax[1].set_ylabel('Frequency')
        fig.suptitle('Distribution Words in Disaster Tweets (target = 1)')
        ax[0].figure.savefig("./plots/distribution-words1.png")
        #plt.show()

        """ Distribution of words in non-disaster tweets (target = 0) """

        sns.set(rc={'figure.figsize':(12,5)})
        fig, ax = plt.subplots(1, 2)

        text_len1 = train_df[train_df['target'] == 0]['text'].str.split().map(lambda x: len(x))
        ax[0].hist(text_len1, color = "yellowgreen")
        ax[0].set_title('Before preprocessing')
        ax[0].set_xlabel('Number of Words')
        ax[0].set_ylabel('Frequency')

        text_len2 = train_df_new[train_df_new['target'] == 0]['new_text'].str.split().map(lambda x: len(x))
        ax[1].hist(text_len2, color = "yellowgreen")
        ax[1].set_title('After preprocessing')
        ax[1].set_xlabel('Number of Words')
        ax[1].set_ylabel('Frequency')
        fig.suptitle('Distribution Words in Non-Disaster Tweets (target = 0)')
        ax[0].figure.savefig("./plots/distribution-words0.png")
        #plt.show()
        print('Data Has Been Fully Preprocessed!')
        return train_df_new


    def test_data_preprocessing(self):
        print('============= Initial Data Loading and Preprocessing =============')

        """ .csv test dataset load """
        
        test_df = pd.read_csv(self.test_csv_dir)

        """ Replace NAs """

        test_df['location'] = test_df['location'].fillna('None')
        test_df['keyword'] = test_df['keyword'].fillna('None')
        print('<<< N/A values filled >>>')

        """ Lowercasing """

        test_df['text'] = test_df['text'].apply(lambda x: x.lower())
        print('<<< Lowercased >>>')

        """ Basic Cleaning - tokenization, URLs, digits, punctuations, non-ascii, emails removal """

        def spacy_clean(text):
            preprocessed = []
            doc = nlp(text)
            preprocessed = [token.lemma_ for token in doc if not token.is_punct and not token.is_digit and not token.like_url and not token.like_email and token.is_ascii]
            return preprocessed

        test_df['new_text'] = test_df['text'].apply(spacy_clean)
        test_df['new_text'] = [' '.join(map(str, l)) for l in test_df['new_text']]
        print('<<< Tokenized, URLs, digits, punctuations, non-ascii, emails removed >>>')
        test_df['new_text'] = test_df["new_text"].str.replace('[^\w\s]', "").str.replace('[0-9]', "").str.replace(' [a-z] ', "").str.replace('-', "").str.replace('_', "").str.replace('&amp', "").str.replace('@', "")
        print('<<< Special characters and numbers removed >>>')

        """ Advanced Cleaning - tokenization, lemmanization, stop words, URLs, digits, punctuations, non-ascii, emails removal """

        #def spacy_clean(text):
            #preprocessed = []
            #doc = nlp(text)
            #preprocessed = [token.lemma_ for token in doc if not token.is_stop and not nlp.vocab[token.lemma_].is_stop and not token.is_punct and not token.is_digit and not token.like_url and not token.like_email and token.is_ascii]
            #return preprocessed

        #test_df['new_text'] = test_df['text'].apply(spacy_clean)
        #test_df['new_text'] = [' '.join(map(str, l)) for l in test_df['new_text']]
        #print('<<< Tokenized, lemmanized, stop words, URLs, digits, punctuations, non-ascii, emails removed >>>')
        #test_df['new_text'] = test_df["new_text"].str.replace('[^\w\s]', "").str.replace('[0-9]', "").str.replace(' [a-z] ', "").str.replace('-', "").str.replace('_', "").str.replace(' amp ', "").str.replace('@', "")
        #print('<<< Special characters and numbers removed >>>')
        #test_df.head()
        return test_df
