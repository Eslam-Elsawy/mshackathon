from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
import pandas as pd
import os
from django.conf import settings
import numpy as np
import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble  import RandomForestClassifier as RandomForestClassifierAlgo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB as MultinomialNBAlgo
from sklearn.ensemble import ExtraTreesClassifier as ExtraTreesClassifierAlgo
from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifierAlgo
from sklearn.linear_model import LogisticRegression as LogisticRegressionAlgo

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

mlb = None
RandomForestClassifier = None
ExtraTreesClassifier = None
DecisionTreeClassifier = None
LogisticRegression = None
NB = None

def home(request):
    global mlb
    global RandomForestClassifier
    global ExtraTreesClassifier
    global DecisionTreeClassifier
    global LogisticRegression
    global NB
    if not RandomForestClassifier:

        stop_words = set(stopwords.words('english'))

        categories = ['Ease of use', 'Love it', 'Good', 'Hard to use', 'Useful', 'Pricing', 'Slow', 'Sharing',
                      'Feature Gap', 'Buggy', 'Too early', 'Refresh', 'Navigation', 'Foreign Language', 'Visuals',
                      'Export',
                      'Desktop', 'Mobile', 'filters', 'Product updates', 'Connectors', 'UI', 'Performance (bad)',
                      'Documentation', 'NPS dialogue', 'Excel', 'Report Creation', 'Accuracy', 'Update', 'Insight',
                      'Fast',
                      'Tableau', 'UX', 'Print', 'Support', 'Competition', 'Training', 'Ease to use', 'licensing',
                      'Access control', 'Flexibility', 'Hate it', 'Report', 'Gateway', 'Dashboard Creation',
                      'Drill down',
                      'Consumption', 'DAX', 'Customization', 'Collaboration', 'PBI Desktop', 'Data Cleansing',
                      'not good',
                      'complicated', 'Performance (Good)', 'Funny', 'iOS', 'Administration', 'not working', 'Helpful',
                      'MacOS', 'Community', 'SharePoint', 'feature', 'Q&amp;A']

        print("reading training data ...")
        trainingdata_filename = os.path.join(settings.BASE_DIR, 'mshackathon/static/TaggedData.csv')

        df = pd.read_csv(trainingdata_filename, encoding="ISO-8859-1")
        print("Training data size: " + str(len(df.values)))

        # ----------------
        # get comments tags
        comments = {}
        count = 0
        for index, row in df.iterrows():
            comment = row['Comment']
            tag = row['Tag']

            if not comment or not tag or tag not in categories:
                continue
            count = count + 1

            if comment in comments:
                commentTags = comments[comment]
                commentTags.append(tag)
                comments[comment] = commentTags
            else:
                comments[comment] = [tag]
        print("Data with good labels:" + str(count))

        # -----------
        # split test and train dat
        print("split test and train dataset")
        DataX_list = []
        DataY_list = []

        for comment in comments:
            DataX_list.append(comment)
            DataY_list.append((comments[comment]))

        DataX = np.array(DataX_list)
        DataY = np.array(DataY_list)

        # # ---------------------
        # multi label binarization
        mlb = MultiLabelBinarizer()
        DataYmlb = mlb.fit_transform(DataY)

        # # training classifiers
        # classifier 1 - RandomForestClassifier
        RandomForestClassifier = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            ('clf', RandomForestClassifierAlgo()),
        ])

        # classifier 2 - ExtraTreesClassifier
        ExtraTreesClassifier = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            ('clf', ExtraTreesClassifierAlgo()),
        ])

        # classifier 3 - DecisionTreeClassifier
        DecisionTreeClassifier = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            ('clf', DecisionTreeClassifierAlgo()),
        ])

        # classifier 4 - LogisticRegression
        LogisticRegression = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            ('clf', OneVsRestClassifier(LogisticRegressionAlgo(solver='sag'), n_jobs=1)),
        ])

        # classifier 5 - NB
        NB = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            ('clf', OneVsRestClassifier(MultinomialNBAlgo(fit_prior=True, class_prior=None))),
        ])

        print("traing class RandomForestClassifier")
        RandomForestClassifier.fit(DataX, DataYmlb)
        print("traing class ExtraTreesClassifier")
        ExtraTreesClassifier.fit(DataX, DataYmlb)
        print("traing class DecisionTreeClassifier")
        DecisionTreeClassifier.fit(DataX, DataYmlb)
        print("traing class LogisticRegression")
        LogisticRegression.fit(DataX, DataYmlb)
        print("training class NB")
        NB.fit(DataX, DataYmlb)

    else:
        print("classifiers already loaded ....")

    return render(request, 'home.html')


@csrf_exempt
def analyze(request):
    json_data = json.loads(request.body)
    comment = json_data['text']
    print("Comment: " + json_data['text'])

    classifiers = [RandomForestClassifier, ExtraTreesClassifier,
                   DecisionTreeClassifier, LogisticRegression, NB]
    classifiers_names = ['Random Forest Classifier', 'Extra Trees Classifier',
                   'Decision Tree Classifier', 'Logistic Regression Classifier',
                         'Naive Bayes Classifier']
    # predict
    prediction = ''
    for classifier, classifiers_name in zip(classifiers, classifiers_names):
        print(classifiers_name)
        predicted = classifier.predict([comment])
        predicted_labels = mlb.inverse_transform(predicted)
        predicted_labels_formatted = ",".join(predicted_labels[0])
        prediction_str = classifiers_name + ": " + predicted_labels_formatted
        prediction = prediction + prediction_str + "\n"
    print(prediction)

    return HttpResponse(prediction)