import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


raw_filename = 'raw_data.csv'

model_registered = [
        {'SVM-linear': svm.LinearSVC(C=0.1, class_weight='balanced')},
        {'SVM-RBF': svm.SVC(C=1, class_weight='balanced',kernel='rbf', tol=1e-10)},
        {'Logistic_Regression': LogisticRegression(solver='sag')},
        {'Random Forest': RandomForestClassifier(n_estimators=200, criterion='entropy',max_features='auto')}
        ]



def text_preprocess(raw_data):
    news_df = raw_data.iloc[:, :-1].copy() # extract the news titles
    
    # Remove the punctuations
    news_df = news_df.replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)
    # Convert the titles to lower cases
    news_df = news_df.applymap(str.lower)
    # Combine all the titles for each date to form a series of contents
    contents = [' '.join(list(news_df.iloc[i, :])) for i in range(len(news_df))]
    # Replace multiple spaces with single space
    contents = [' '.join(content.split()) for content in contents]
    contents = pd.DataFrame(contents, index=raw_data.index, columns=['Content'])
    
    # Incorporate the news_df to the original dataframe
    new_data = contents.copy()
    new_data['Label'] = raw_data.iloc[:, -1]
    
    return new_data


def split_train_and_test(whole_df, train_pct=0.8):
    train_size = int(len(whole_df) * train_pct)
    
    train_df = whole_df.iloc[:train_size, :]
    test_df = whole_df.iloc[train_size:, :]
    
    return train_df, test_df


def split_X_and_Y(df, label_colname):
    train_cols = list(df.columns)
    train_cols.remove(label_colname)
    X = df[train_cols].values
    X = [item[0] for item in X]
    Y = df[label_colname].values
    return X, Y


def text_vectorizer(txt_data, n_gram):
    vectorizer = CountVectorizer(ngram_range=(n_gram, n_gram))
    train_vec = vectorizer.fit_transform(txt_data)
    return train_vec


if __name__ == '__main__':
    
    raw_df = pd.read_csv(raw_filename, index_col=0)
    
    new_df = text_preprocess(raw_df)

    train_df, test_df = split_train_and_test(new_df)
    trainX, trainY = split_X_and_Y(train_df, label_colname='Label')
    testX, testY = split_X_and_Y(test_df, label_colname='Label')
    

    #n_grams = [1, 2, 3]
    n_grams = [3]
    for n_gram in n_grams:
        # Built n-gram model, and vectorize trainX and testX
        #vectorizer = CountVectorizer(ngram_range=(n_gram, n_gram), stop_words='english', max_df=0.7)
        vectorizer = CountVectorizer(ngram_range=(1, n_gram), stop_words='english', max_df=0.7)
        trainX_vec = vectorizer.fit_transform(trainX)
        testX_vec = vectorizer.transform(testX)
        
        print('\n\n\n=========== {}-gram ==========\n'.format(n_gram))
        for model_dict in model_registered:
            classifier, model = list(model_dict.items())[0]
            
            model = model.fit(trainX_vec, trainY)
            pred = model.predict(testX_vec)
            
            print('\n\n*** Model: {} ***\n'.format(classifier))
            two_way_table = pd.crosstab(testY, pred, rownames=['Actual'], colnames=['Predicted'])
            #classfy_report = classification_report(testY, pred)
            accuracy = accuracy_score(testY, pred)
            
            print(two_way_table)
            #print('\n\nClassification Report:')
            #print(classfy_report)
            print('\nAccuracy: {}'.format(accuracy))
