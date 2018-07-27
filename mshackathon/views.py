from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
import pandas as pd
import os
from django.conf import settings
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble  import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

classifier = None
testsize = 0.1
mlb = None

def home(request):
    global classifier
    global testsize
    global mlb
    if not classifier:
        print("loading classifier")
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

        X_train, X_test, Y_train, Y_test = train_test_split(DataX, DataY, random_state=42, test_size=testsize)

        # ---------------------
        # multi label binarization
        mlb = MultiLabelBinarizer()
        Y_train_mlb = mlb.fit_transform(Y_train)

        # classifier
        print("Training classifier")
        classifier = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            ('clf', RandomForestClassifier()),
        ])
        classifier.fit(X_train, Y_train_mlb)

    else:
        print("classifiers already loaded ....")

    return render(request, 'home.html')


@csrf_exempt
def analyze(request):
    json_data = json.loads(request.body)
    comment = json_data['text']
    print("Comment: " + json_data['text'])

    # predict
    predicted = classifier.predict([comment])
    predicted_labels = mlb.inverse_transform(predicted)
    predicted_labels_formatted = ",".join(predicted_labels[0])
    print("Predicted Labels: " + predicted_labels_formatted)
    print(predicted_labels_formatted)

    return HttpResponse(predicted_labels_formatted)