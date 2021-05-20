# import libraries
import io
import csv
import re
import os
import keras
import tweepy
import random
# import pdfkit
import numpy as np
import pandas as pd
from tweepy import OAuthHandler
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from flask import Flask, make_response, request, session, redirect, url_for, escape, current_app
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv(r'telecom_reviews_with_neutral.csv', encoding='unicode_escape')
data = data[['review', 'sentiment']]

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['review'].values)

# create OAuthHandler object
auth = OAuthHandler('uGPmwnvUHzB3tKZrB8KO3AHSU', 'rsrC4yiAIg4Bd1yXlsWrM536VFmkusgNFmhBiXALcl1IU5hoA7')
# set access token and secret
auth.set_access_token('1321517957348876289-EBnDqUJvgrj8Lx0SOzd56OZbflpmkf',
                      'o8sgu7ha9iDBkvnv33F76CePHscEUsBgbFwNKpEIwfp4g')

# create tweepy API object to fetch tweets
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# load json and create model
json_file = open('modelx2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("modelx2.h5")

# For Categorizing
# load json and create model
cat_json_file = open('model3.json', 'r')
cat_model_json = cat_json_file.read()
cat_json_file.close()
cat_model = model_from_json(cat_model_json)
# load weights into new model
cat_model.load_weights("model3.h5")

TAG_RE = re.compile(r'<[^>]+>')

# Initialize the flask App
app = Flask(__name__)
app.secret_key = 'secret key twitter'


# preprocessing
def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    # Removing links
    sentence = re.sub(r'^https?:\/\/.*[\r\n]*', '', sentence, flags=re.MULTILINE)

    # Remove words starting with special characters
    sentence = re.sub(r'(\s)#\w+', r'\1', sentence)

    return sentence


def remove_tags(text):
    return TAG_RE.sub('', text)


@app.route('/')
def login():
    return render_template('login.html')


@app.route('/', methods=['POST'])
def redirect_to_home():
    if request.method == 'POST':
        session['username'] = request.form['username']
    return render_template('home.html', user=session['username'])


@app.route('/contact')
@app.route('/help')
@app.route('/home')
def home():
    return render_template('home.html', user=session['username'])


@app.route('/twitter')
def tweets():
    return render_template('twitter.html')


# For Twitter Data
@app.route('/twitter', methods=['POST'])
def predict_tweets():
    cat1 = 0
    cat2 = 0
    cat3 = 0
    cat4 = 0
    poscnt = 0
    negcnt = 0
    tweetcnt = 0
    cat1_negcnt = 0
    cat1_poscnt = 0
    cat2_negcnt = 0
    cat2_poscnt = 0
    cat3_negcnt = 0
    cat3_poscnt = 0
    cat4_negcnt = 0
    cat4_poscnt = 0
    cnt = 0
    n = random.randint(1, 100)

    if request.method == 'POST':
        search = session['username']
        fromdate = request.form['fromdate']
        todate = request.form['todate']
        n_tweets = []
        n_ops = []
        p_tweets = []
        p_ops = []
        tweets_cat1 = []
        ops_cat1 = []
        tweets_cat2 = []
        ops_cat2 = []
        tweets_cat3 = []
        ops_cat3 = []
        tweets_cat4 = []
        ops_cat4 = []

        new_search = search + " -filter:retweets"

        date_since = str(fromdate)
        date_until = str(todate)
        print(date_since)
        print(date_until)

        search_results = tweepy.Cursor(api.search,
                                       q=new_search,
                                       lang="en",
                                       since=date_since,
                                       until=date_until,
                                       exclude_replies=True,
                                       include_entities=False).items()

        for tweet in search_results:
            cnt += 1
            if (tweet.user.screen_name != search.lower()) and (not tweet.retweeted) and ('RT @' not in tweet.text):
                print(tweet.created_at)
                print(tweet.user.screen_name)
                cleaned_tweet = preprocess_text(tweet.text)
                x = []
                x.clear()
                x.append(cleaned_tweet)
                w = tokenizer.texts_to_sequences(x)
                twt1 = pad_sequences(w, maxlen=174, dtype='int32', value=0)
                op = loaded_model.predict(twt1)
                category = cat_model.predict(twt1)

                if (np.argmax(op) == 0):
                    tweetcnt = tweetcnt + 1
                    negcnt = negcnt + 1
                    n_tweets.append(tweet.text)
                    n_ops.append('Negative')
                    print(cleaned_tweet)
                    print(': Neg')
                elif (np.argmax(op) == 2):
                    tweetcnt = tweetcnt + 1
                    poscnt = poscnt + 1
                    p_tweets.append(tweet.text)
                    p_ops.append('Positive')
                    print(cleaned_tweet)
                    print(': Pos')
                elif(np.argmax(op) == 1):
                    print(cleaned_tweet)
                    print(': Neu')

                # categories
                if (np.argmax(op) == 0 or np.argmax(op) == 2):
                    if (np.argmax(category) == 0):
                        cat1 = cat1 + 1
                        tweets_cat1.append(tweet.text)
                        ops_cat1.append('4G/ Internet Homebroadband')
                        if (np.argmax(op) == 0):
                            cat1_negcnt = cat1_negcnt + 1
                        elif (np.argmax(op) == 2):
                            cat1_poscnt = cat1_poscnt + 1
                    elif (np.argmax(category) == 1):
                        cat2 = cat2 + 1
                        tweets_cat2.append(tweet.text)
                        ops_cat2.append('Network/ Coverage')
                        if (np.argmax(op) == 0):
                            cat2_negcnt = cat2_negcnt + 1
                        elif (np.argmax(op) == 2):
                            cat2_poscnt = cat2_poscnt + 1
                    elif (np.argmax(category) == 2):
                        cat3 = cat3 + 1
                        tweets_cat3.append(tweet.text)
                        ops_cat3.append('Customer Service')
                        if (np.argmax(op) == 0):
                            cat3_negcnt = cat3_negcnt + 1
                        elif (np.argmax(op) == 2):
                            cat3_poscnt = cat3_poscnt + 1
                    elif (np.argmax(category) == 3):
                        cat4 = cat4 + 1
                        tweets_cat4.append(tweet.text)
                        ops_cat4.append('Other Matters')
                        if (np.argmax(op) == 0):
                            cat4_negcnt = cat4_negcnt + 1
                        elif (np.argmax(op) == 2):
                            cat4_poscnt = cat4_poscnt + 1

        print(cnt)

    negoutput = dict(zip(n_tweets, n_ops))
    posoutput = dict(zip(p_tweets, p_ops))
    output_cat1 = dict(zip(tweets_cat1, ops_cat1))
    output_cat2 = dict(zip(tweets_cat2, ops_cat2))
    output_cat3 = dict(zip(tweets_cat3, ops_cat3))
    output_cat4 = dict(zip(tweets_cat4, ops_cat4))
    data = {'Task': 'Sentiment Analysis', 'Positive': poscnt, 'Negative': negcnt}
    Internet = {'Task': 'Internet', 'Positive': cat1_poscnt, 'Negative': cat1_negcnt}
    Network = {'Task': 'Network', 'Positive': cat2_poscnt, 'Negative': cat2_negcnt}
    Customer = {'Task': 'Customer', 'Positive': cat3_poscnt, 'Negative': cat3_negcnt}
    Other = {'Task': 'Other', 'Positive': cat4_poscnt, 'Negative': cat4_negcnt}
    return render_template('pie-chart-twitter.html', data=data, negoutputs=negoutput, posoutputs=posoutput,
                           output_cat1=output_cat1, output_cat2=output_cat2, output_cat3=output_cat3,
                           output_cat4=output_cat4, Internet=Internet, Network=Network, Customer=Customer, Other=Other,
                           tweetcnt=tweetcnt, cat1=cat1, cat2=cat2, cat3=cat3, cat4=cat4)


@app.route('/user', methods=['GET'])
def user():
    return render_template('user.html')


# For User Data
# To use the predict button in the web-app
@app.route('/user', methods=['POST'])
def predict():
    cat1 = 0
    cat2 = 0
    cat3 = 0
    cat4 = 0
    poscnt = 0
    negcnt = 0
    tweetcnt = 0
    cat1_negcnt = 0
    cat1_poscnt = 0
    cat2_negcnt = 0
    cat2_poscnt = 0
    cat3_negcnt = 0
    cat3_poscnt = 0
    cat4_negcnt = 0
    cat4_poscnt = 0

    if request.method == "POST":
        if request.files:
            csv_upload = request.files["files"]
            filename = csv_upload.filename
            csv_upload.save(os.path.join("uploads", csv_upload.filename))
            path = os.path.join("uploads", csv_upload.filename)
            n_tweets = []
            n_ops = []
            p_tweets = []
            p_ops = []
            tweets_cat1 = []
            ops_cat1 = []
            tweets_cat2 = []
            ops_cat2 = []
            tweets_cat3 = []
            ops_cat3 = []
            tweets_cat4 = []
            ops_cat4 = []
            with open(path, encoding='unicode_escape') as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                for row in readCSV:
                    tweetcnt = tweetcnt + 1
                    a = preprocess_text(row[0])
                    x = []
                    x.clear()
                    x.append(row[0])
                    w = tokenizer.texts_to_sequences(x)
                    print(x)
                    print(w)

                    twt1 = pad_sequences(w, maxlen=174, dtype='int32', value=0)

                    sentiment1 = loaded_model.predict(twt1, batch_size=1, verbose=2)[0]
                    category = cat_model.predict(twt1)

                    # sentiment
                    if (np.argmax(sentiment1) == 0):
                        negcnt = negcnt + 1
                        n_tweets.append(row[0])
                        n_ops.append('Negative')
                    elif (np.argmax(sentiment1) == 2):
                        poscnt = poscnt + 1
                        p_tweets.append(row[0])
                        p_ops.append('Positive')

                    # categories
                    if (np.argmax(sentiment1) == 0 or np.argmax(sentiment1) == 2):
                        if (np.argmax(category) == 0):
                            cat1 = cat1 + 1
                            tweets_cat1.append(row[0])
                            ops_cat1.append('4G/ Internet Homebroadband')
                            if (np.argmax(sentiment1) == 0):
                                cat1_negcnt = cat1_negcnt + 1
                            elif (np.argmax(sentiment1) == 2):
                                cat1_poscnt = cat1_poscnt + 1
                        elif (np.argmax(category) == 1):
                            cat2 = cat2 + 1
                            tweets_cat2.append(row[0])
                            ops_cat2.append('Network/ Coverage')
                            if (np.argmax(sentiment1) == 0):
                                cat2_negcnt = cat2_negcnt + 1
                            elif (np.argmax(sentiment1) == 2):
                                cat2_poscnt = cat2_poscnt + 1
                        elif (np.argmax(category) == 2):
                            cat3 = cat3 + 1
                            tweets_cat3.append(row[0])
                            ops_cat3.append('Customer Service')
                            if (np.argmax(sentiment1) == 0):
                                cat3_negcnt = cat3_negcnt + 1
                            elif (np.argmax(sentiment1) == 2):
                                cat3_poscnt = cat3_poscnt + 1
                        elif (np.argmax(category) == 3):
                            cat4 = cat4 + 1
                            tweets_cat4.append(row[0])
                            ops_cat4.append('Other Matters')
                            if (np.argmax(sentiment1) == 0):
                                cat4_negcnt = cat4_negcnt + 1
                            elif (np.argmax(sentiment1) == 2):
                                cat4_poscnt = cat4_poscnt + 1

    negoutput = dict(zip(n_tweets, n_ops))
    posoutput = dict(zip(p_tweets, p_ops))
    output_cat1 = dict(zip(tweets_cat1, ops_cat1))
    output_cat2 = dict(zip(tweets_cat2, ops_cat2))
    output_cat3 = dict(zip(tweets_cat3, ops_cat3))
    output_cat4 = dict(zip(tweets_cat4, ops_cat4))
    data = {'Task': 'Sentiment Analysis', 'Positive': poscnt, 'Negative': negcnt}
    Internet = {'Task': 'Internet', 'Positive': cat1_poscnt, 'Negative': cat1_negcnt}
    Network = {'Task': 'Network', 'Positive': cat2_poscnt, 'Negative': cat2_negcnt}
    Customer = {'Task': 'Customer', 'Positive': cat3_poscnt, 'Negative': cat3_negcnt}
    Other = {'Task': 'Other', 'Positive': cat4_poscnt, 'Negative': cat4_negcnt}
    return render_template('pie-chart-user.html', data=data, negoutputs=negoutput, posoutputs=posoutput,
                           output_cat1=output_cat1, output_cat2=output_cat2, output_cat3=output_cat3,
                           output_cat4=output_cat4, Internet=Internet, Network=Network, Customer=Customer, Other=Other,
                           tweetcnt=tweetcnt, cat1=cat1, cat2=cat2, cat3=cat3, cat4=cat4)


if __name__ == "__main__":
    app.run(debug=True)
