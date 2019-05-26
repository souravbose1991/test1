import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import sys
import os, re, math
from collections import Counter
from pandas.compat import StringIO


global pathname
pathname = '/home/sourav/Deloitte'



def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def expand_contractions(text):
    return contractions.fix(text)


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text


def pre_process_document(document):

	import contractions
	from bs4 import BeautifulSoup
	import unicodedata
	import re
	
	# nltk.download('wordnet')

	# NLTK
	import nltk
	from nltk.corpus import stopwords
	from nltk.stem import SnowballStemmer
	from nltk.stem import WordNetLemmatizer
	from nltk.tokenize import RegexpTokenizer 
	stop_words = set(stopwords.words("english")) 
    
    # strip HTML
    document = strip_html_tags(document)
    
    # lower case
    document = document.lower()
    
    # remove extra newlines (often might be present in really noisy text)
    document = document.translate(document.maketrans("\n\t\r", "   "))
    
    # remove accented characters
    document = remove_accented_chars(document)
    
    # expand contractions    
    document = expand_contractions(document)
               
    # remove special characters and\or digits    
    # insert spaces between special characters to isolate them    
    special_char_pattern = re.compile(r'([{.(-)!}])')
    document = special_char_pattern.sub(" \\1 ", document)
    document = remove_special_characters(document, remove_digits=True)  
        
    # remove extra whitespace
    document = re.sub(' +', ' ', document)
    document = document.strip()
    
    document = document.split()
    lemmatizer = WordNetLemmatizer()
    document = [lemmatizer.lemmatize(word, pos='v') for word in document]
    document = [word for word in document if not word in stop_words]
    document = " ".join(document)
    return document
	

	
# MODEL EVALUATION METRICES

def eval1(actual,predicted):
    acc = accuracy_score(actual,predicted)
    rec = recall_score(actual,predicted, average='weighted')
    pre = precision_score(actual,predicted, average='weighted')
    f1 = f1_score(actual,predicted, average='weighted')
    print("Accuracy: \t"+ str(acc))
    print("Recall: \t" + str(rec))
    print("Precision: \t" + str(pre))
    print("F1 Score: \t" + str(f1))
    return(acc, rec, pre, f1)

   

# change the kfold to higher number if you want to do cross validation- by default it is 1
def classify(df, df_test, ws, lr, dim, wordNgrams, epoch, kfold=1):  

	import fastText as ft
	from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
	from sklearn.model_selection import ParameterGrid, train_test_split, KFold
    
#     df['human_label'] = df.human_label.astype('category').cat.codes
    df['labels_text'] = '__label__' + df['Sentiment'].astype(str)
    df = df[['Sentiment','Reviews', 'labels_text']]
    df.labels_text = df.labels_text.str.cat(df.Reviews, sep=' ')

    if kfold > 1:        
        acc_ar = np.array([])
        rec_ar = np.array([])
        pre_ar = np.array([])
        f1_ar = np.array([])
		
		acc_ar2 = np.array([])
        rec_ar2 = np.array([])
        pre_ar2 = np.array([])
        f1_ar2 = np.array([])
        
        kf = KFold(n_splits=kfold, random_state=42, shuffle=True)
        temp_fold = 1
        for train_index, test_index in kf.split(df):
            dum_train = df.iloc[train_index]
            dum_test = df.iloc[test_index]
    
            # Write the training data for this CV fold into text file
            train_cv_file = open(pathname +'/FastText/train_cv.txt','w')
            train_cv_file.writelines(dum_train.labels_text + '\n')
            train_cv_file.close()
            Train_cv_path=pathname+'/FastText/train_cv.txt'
        
            classifier = ft.FastText.train_supervised(Train_cv_path, ws=ws, lr=lr, dim=dim, wordNgrams=wordNgrams, epoch=epoch)
            predictions = classifier.predict(dum_train.Reviews.tolist(),k=1,threshold=0.0)
            pred_class = np.array([])
            pred_score = np.array([])
            
            for i in range(len(predictions[0])):
                pred_class = np.append(pred_class,predictions[0][i][0])
                pred_score = np.append(pred_score, predictions[1][i][0])

            pred_class = pd.Series(pred_class).apply(lambda x: re.sub('__label__', '', x))
            print("\nModel Evaluation Measurement for Fold=" + str(temp_fold) + " :")
            t_acc, t_rec, t_pre, t_f1 = eval1(dum_train.Sentiment,pred_class)

            acc_ar = np.append(acc_ar, t_acc)
            rec_ar = np.append(rec_ar, t_rec)
            pre_ar = np.append(pre_ar, t_pre)
            f1_ar = np.append(f1_ar, t_f1)
			
			
			
			predictions = classifier.predict(dum_test.Reviews.tolist(),k=1,threshold=0.0)
            pred_class = np.array([])
            pred_score = np.array([])
            
            for i in range(len(predictions[0])):
                pred_class = np.append(pred_class,predictions[0][i][0])
                pred_score = np.append(pred_score, predictions[1][i][0])

            pred_class = pd.Series(pred_class).apply(lambda x: re.sub('__label__', '', x))
            print("\nModel Evaluation Measurement for Fold=" + str(temp_fold) + " :")
            t_acc, t_rec, t_pre, t_f1 = eval1(dum_test.Sentiment,pred_class)

            acc_ar2 = np.append(acc_ar2, t_acc)
            rec_ar2 = np.append(rec_ar2, t_rec)
            pre_ar2 = np.append(pre_ar2, t_pre)
            f1_ar2 = np.append(f1_ar2, t_f1)
            
            temp_fold+=1
        
        acc = np.mean(acc_ar)
        rec = np.mean(rec_ar)
        pre = np.mean(pre_ar)
        f1 = np.mean(f1_ar)
		
		acc2 = np.mean(acc_ar2)
        rec2 = np.mean(rec_ar2)
        pre2 = np.mean(pre_ar2)
        f12 = np.mean(f1_ar2)
		
		s = ((acc,rec,pre,f1),(acc2,rec2,pre2,f12))
		
		df_sb = pd.DataFrame(list(s), columns=['acc','rec','pre','f1'], index=['train','test'])

        print("\nFinal Aggregated Evaluation Metrics:")
        print("Accuracy: \t"+ str(acc2))
        print("Recall: \t" + str(rec2))
        print("Precision: \t" + str(pre2))
        print("F1-Score: \t"+ str(f12))
        
    else:

        training_file = open(pathname + '/FastText/train.txt','w')
        training_file.writelines(df.labels_text + '\n')
        training_file.close()
        Train_path=pathname + '/FastText/train.txt'
        
        classifier = ft.FastText.train_supervised(Train_path, ws=ws, lr=lr, dim=dim, wordNgrams=wordNgrams, epoch=epoch)
        classifier.save_model(pathname + '/FastText/FastText_model.ftz')
        
        predictions = classifier.predict(df.Reviews.tolist(),k=1,threshold=0.0)
        pred_class = np.array([])
        pred_score = np.array([])

        for i in range(len(predictions[0])):
            pred_class = np.append(pred_class,predictions[0][i][0])
            pred_score = np.append(pred_score, predictions[1][i][0])

        pred_class = pd.Series(pred_class).apply(lambda x: re.sub('__label__', '', x))
        acc, rec, pre, f1 = eval1(df.Sentiment, pred_class)
		
		
		predictions = classifier.predict(df_test.Reviews.tolist(),k=1,threshold=0.0)
        pred_class = np.array([])
        pred_score = np.array([])

        for i in range(len(predictions[0])):
            pred_class = np.append(pred_class,predictions[0][i][0])
            pred_score = np.append(pred_score, predictions[1][i][0])

        pred_class = pd.Series(pred_class).apply(lambda x: re.sub('__label__', '', x))
        acc2, rec2, pre2, f12 = eval1(df_test.Sentiment, pred_class)
		
		s = ((acc,rec,pre,f1),(acc2,rec2,pre2,f12))
		df_sb = pd.DataFrame(list(s), columns=['acc','rec','pre','f1'], index=['train','test'])
        
        print("\nFinal Evaluation Metrics:")
        print("Accuracy: \t"+ str(acc2))
        print("Recall: \t" + str(rec2))
        print("Precision: \t" + str(pre2))
        print("F1-Score: \t"+ str(f12))
		
    return(df_sb)	
	
	
	
def combo_ft(df, ws, lr, dim, wordNgrams, epoch):

	from sklearn.model_selection import train_test_split
	
	df2 = df
	df2.Reviews = df1.Reviews.apply(lambda x: pre_process_document(x))	
	daf, daf_test = train_test_split(df2, test_size=0.2, random_state =42)
	df_sb = classify(df=daf, df_test=daf_test, ws=ws, lr=lr, dim=dim, wordNgrams=wordNgrams, epoch=epoch, kfold=1)
	return(df_sb)
	
	
	
def combo_rf(df, n_estimators, max_depth, criterion, max_features):

	from sklearn.preprocessing import LabelEncoder
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.feature_extraction.text import TfidfTransformer
	from sklearn.model_selection import train_test_split
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
	
	df2 = df
	df2.Reviews = df1.Reviews.apply(lambda x: pre_process_document(x))
	tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 3), stop_words='english')
	features = tfidf.fit_transform(df2['Reviews'].values)
	labels = LabelEncoder().fit_transform(df2['Sentiment'])

	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
	clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, max_features=max_features, random_state=42)
	clf.fit(X_train, y_train)  
	
	y_pred = clf.predict(X_train)
	acc = accuracy_score(y_train, y_pred)
	rec = recall_score(y_train, y_pred)
	pre = precision_score(y_train, y_pred)
	f1 = f1_score(y_train, y_pred)
	
	y_pred = clf.predict(X_test)
	acc2 = accuracy_score(y_test, y_pred)
	rec2 = recall_score(y_test, y_pred)
	pre2 = precision_score(y_test, y_pred)
	f12 = f1_score(y_test, y_pred)
	
	s = ((acc,rec,pre,f1),(acc2,rec2,pre2,f12))
	df_sb = pd.DataFrame(list(s), columns=['acc','rec','pre','f1'], index=['train','test'])
	return(df_sb)
	
	
	
def combo_lsvc(df, penalty , C, loss, max_iter):
	
	from sklearn.preprocessing import LabelEncoder
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.feature_extraction.text import TfidfTransformer
	from sklearn.model_selection import train_test_split
	from sklearn.svm import LinearSVC
	from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
	
	df2 = df
	df2.Reviews = df1.Reviews.apply(lambda x: pre_process_document(x))
	tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 3), stop_words='english')
	features = tfidf.fit_transform(df2['Reviews'].values)
	labels = LabelEncoder().fit_transform(df2['Sentiment'])

	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
	clf = LinearSVC(penalty =penalty, C=C, loss=loss, max_iter=max_iter, random_state=42)
	clf.fit(X_train, y_train)  
	
	y_pred = clf.predict(X_train)
	acc = accuracy_score(y_train, y_pred)
	rec = recall_score(y_train, y_pred)
	pre = precision_score(y_train, y_pred)
	f1 = f1_score(y_train, y_pred)
	
	y_pred = clf.predict(X_test)
	acc2 = accuracy_score(y_test, y_pred)
	rec2 = recall_score(y_test, y_pred)
	pre2 = precision_score(y_test, y_pred)
	f12 = f1_score(y_test, y_pred)
	
	s = ((acc,rec,pre,f1),(acc2,rec2,pre2,f12))
	df_sb = pd.DataFrame(list(s), columns=['acc','rec','pre','f1'], index=['train','test'])
	return(df_sb)
	
	
	
	
def combo_lr(df, multi_class, solver, penalty ,C, max_iter):
	
	from sklearn.preprocessing import LabelEncoder
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.feature_extraction.text import TfidfTransformer
	from sklearn.model_selection import train_test_split
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
	
	df2 = df
	df2.Reviews = df1.Reviews.apply(lambda x: pre_process_document(x))
	tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 3), stop_words='english')
	features = tfidf.fit_transform(df2['Reviews'].values)
	labels = LabelEncoder().fit_transform(df2['Sentiment'])

	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
	clf = LogisticRegression(multi_class=multi_class, solver=solver, penalty=penalty, C=C, max_iter=max_iter, random_state=42)
	clf.fit(X_train, y_train)  
	
	y_pred = clf.predict(X_train)
	acc = accuracy_score(y_train, y_pred)
	rec = recall_score(y_train, y_pred)
	pre = precision_score(y_train, y_pred)
	f1 = f1_score(y_train, y_pred)
	
	y_pred = clf.predict(X_test)
	acc2 = accuracy_score(y_test, y_pred)
	rec2 = recall_score(y_test, y_pred)
	pre2 = precision_score(y_test, y_pred)
	f12 = f1_score(y_test, y_pred)
	
	s = ((acc,rec,pre,f1),(acc2,rec2,pre2,f12))
	df_sb = pd.DataFrame(list(s), columns=['acc','rec','pre','f1'], index=['train','test'])
	return(df_sb)
	
	
	
	