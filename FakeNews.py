# Import necessary libraries
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from textstat import flesch_kincaid_grade, gunning_fog
import string
from textblob import TextBlob
import spacy

# Load spaCy for NER and POS tagging
nlp = spacy.load('en_core_web_sm')

def DetectNews(input_text):
    # Assuming the existence of the dataset (news_df) and other preprocessing steps
    # If your original code is different, adjust accordingly

    # Assuming news_df is a DataFrame with 'content' and 'label' columns
    # Adjust the column names if needed
    news_df = pd.read_csv("C:\Users\himan\OneDrive\Desktop\project\train.csv")
    news_df = news_df.fillna(' ')
    news_df['content'] = news_df['author'] + " " + news_df['title']

    ps = PorterStemmer()

    def stemming(content):
        stemmed_content = re.sub('[^a-zA-Z]',' ',content)
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.split()
        stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
        stemmed_content = ' '.join(stemmed_content)
        return stemmed_content

    news_df['content'] = news_df['content'].apply(stemming)

    X = news_df['content'].values
    y = news_df['label'].values

    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X)

    news_df['flesch_grade'] = news_df['content'].apply(lambda x: flesch_kincaid_grade(x))
    news_df['gunning_fog'] = news_df['content'].apply(lambda x: gunning_fog(x))
    for punctuation in string.punctuation:
        news_df[f'num_{punctuation}'] = news_df['content'].apply(lambda x: x.count(punctuation))

    # Update this line to include features based on the input_text
    input_length = len(input_text.split())
    input_num_numerical = sum(c.isdigit() for c in input_text)
    input_sentiment = TextBlob(input_text).sentiment.polarity
    input_num_entities = len(nlp(input_text).ents)
    input_num_nouns = len([token for token in nlp(input_text) if token.pos_ == 'NOUN'])

    X_additional = news_df[['flesch_grade', 'gunning_fog'] + [f'num_{p}' for p in string.punctuation]]
    input_features = [input_length, input_num_numerical, input_sentiment, input_num_entities, input_num_nouns]
    input_features += [flesch_kincaid_grade(input_text), gunning_fog(input_text)]
    input_features += [input_text.count(punctuation) for punctuation in string.punctuation]

    # Combine TF-IDF features with additional features for input_text
    input_combined = np.hstack((X_tfidf.transform([input_text]).toarray(), np.array([input_features])))

    X_combined = np.hstack((X_tfidf.toarray(), X_additional.values))

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, stratify=y, random_state=1)

    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    # Predict using the trained model
    prediction = nb_model.predict(input_combined)

    return prediction[0]  # Assuming prediction is either 0 or 1 (REAL or FAKE)

# Test the function
input_news = "House Dem Aide: We Didnâ€™t Even See Comeyâ€™s Letter Until Jason Chaffetz Tweeted It ..."
result = DetectNews(input_news)
print("Result:", result)

