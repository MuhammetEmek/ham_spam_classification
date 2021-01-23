# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 19:16:35 2020

@author: memek
"""

import re
import glob
import string
import itertools
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time 
from pandas import DataFrame
# for preprocessing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# visualization
from wordcloud import WordCloud
# for feature extraction, training and classification
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
# for evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

SPAM = "Spam"
HAM  = "Ham"

COMMON_WORDS = ["subject", "company", "please", "business"]

CLASS_LABELS = [HAM, SPAM]

DATA_FOLDERS = [ 
    ("Data/enron1/ham", HAM ), 
    ("Data/enron1/spam", SPAM ),
    ("Data/enron2/ham", HAM ), 
    ("Data/enron2/spam", SPAM ),
    ("Data/enron3/ham", HAM ), 
    ("Data/enron3/spam", SPAM ),
    ("Data/enron4/ham", HAM ), 
    ("Data/enron4/spam", SPAM ),
    ("Data/enron5/ham", HAM ), 
    ("Data/enron5/spam", SPAM ),
    ("Data/enron6/ham", HAM ), 
    ("Data/enron6/spam", SPAM )]


# stopword list with common words
stop_words = stopwords.words('english') + COMMON_WORDS

def preprocess_data(content_str):    
    # Lemmatization object
    lemmatizer = WordNetLemmatizer()

    # Converting to lowercase
    content_str = content_str.lower()
    
    # Removing punctuation
    content_str = content_str.translate(str.maketrans('','', string.punctuation))
    
    # Removing numbers
    content_str = re.sub(r'\d+', '', content_str)

    # Word lemmatization and Removing stopwords
    content_str = ' '.join([lemmatizer.lemmatize(word) for word in content_str.split() if word not in stop_words])        
  
    # Removing single char words like 'j', 'l'
    content_str = re.sub(r"\b[a-zA-Z]\b", "", content_str)
  
    # Removing whitespaces, tabs and return
    return ' '.join(content_str.split())


def read_class_files(folder_path, class_type):
    print("Reading files under [ " + folder_path + " ] ...")
    rows = []
    index = []
    folder_path = folder_path + "/*.txt"
    file_path_list = glob.glob(folder_path)
    for file_path in file_path_list:
        try:
            with open(file_path,"r", encoding="utf8") as fp:
                email_data = fp.read().replace('\n', '')
                cleaned_file_content = preprocess_data(email_data)
                if cleaned_file_content:
                    rows.append({'email_content': cleaned_file_content, 'class': class_type})
                    index.append(file_path)
        except:
            print("Error occured while reading file on path:" + file_path)
               
    print(class_type + " files loaded")
    data_frame = DataFrame(rows, index=index)
    return data_frame, len(rows) 


def load_dataset():
    start_time = int(time() * 1000)    
    print("Loading data...")
    df = DataFrame({'email_content': [], 'class': []})
    for folder_path, class_type in DATA_FOLDERS:
        data_frame, nrows = read_class_files(folder_path, class_type)
        df = df.append(data_frame)
        
    df = df.reindex(np.random.permutation(df.index))
    print("Data loaded. Total Elapsed Time : %d msn." %(int(time() * 1000)  - start_time))
    return df


def show_term_frequency_with_wordcloud(email_content, description):
    wordcloud = WordCloud(width = 800, height = 800, background_color ='white', min_font_size = 10, collocations=False).generate(email_content) 
    
    with plt.style.context(('ggplot', 'seaborn')):              
        plt.figure(figsize = (8, 8)) 
        plt.imshow(wordcloud, interpolation='bilinear') 
        plt.axis("off") 
        plt.tight_layout(pad = 0)
        plt.title(description)
        plt.show()


def show_term_frequency(content_df, class_type):
    class_data_text_arr = content_df.loc[content_df['class'] == class_type].email_content
    frequency_series = pd.Series(' '.join(class_data_text_arr).split()).value_counts()[:20]
    
    description = "Top 20 Word Frequency For " + class_type
    print("\n" + description)
    print("------------------------------")
    print(frequency_series)
    frequency_series.plot.bar(x='word', y='frequency')
    plt.title(description)
    plt.show()
    
    # show with word cloud
    show_term_frequency_with_wordcloud(' '.join(class_data_text_arr), description)


def show_evaluation_report(evaluation_df):
    print('\n                          Evalutation Report                         ')
    print("***********************************************************************")
    print(evaluation_df)    
    print("***********************************************************************")
          
          
def draw_confusion_matrix(conf_matrix, description):
    with plt.style.context(('ggplot', 'seaborn')):
        plt.imshow(conf_matrix, interpolation='nearest',cmap= plt.cm.Blues )
        plt.xticks([0,1], CLASS_LABELS)
        plt.yticks([0,1], CLASS_LABELS)
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
            plt.text(j, i,conf_matrix[i, j], horizontalalignment="center",color="red")
                
    plt.grid(None)
    plt.title('Confusion Matrix for ' + description)
    plt.colorbar();
    plt.show()
   
def train_and_predict(data, pipeline, classifier_desc, n_folds = 6): 
    start_time = int(time() * 1000)
    
    print("\n****************************************************************")
    print("%s processing..." %classifier_desc)
    print("****************************************************************")
    
    sk_fold = StratifiedKFold(n_splits = n_folds)
    accuracy_scores,precision_scores,recall_scores,f1_scores= [], [], [], []
    conf_matrix = np.array([[0, 0], [0, 0]])
      
    fold_no = 1
    classes = data.loc[:,'class']
    for train_indices, test_indices in sk_fold.split(X=data, y=classes):
        X_train = data.iloc[train_indices]['email_content'].values
        y_train = data.iloc[train_indices]['class'].values.astype(str)

        X_test = data.iloc[test_indices]['email_content'].values
        y_test = data.iloc[test_indices]['class'].values.astype(str)

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)        

        # Collect evaluation values
        conf_matrix += confusion_matrix(y_test, predictions)
        # accuracy: (tp + tn) / (p + n)
        accuracy_scores.append(accuracy_score(y_test, predictions))
        # precision: tp / (tp + fp)
        precision_scores.append(precision_score(y_test, predictions, pos_label=SPAM))
        # recall: tp / (tp + fn)
        recall_scores.append(recall_score(y_test, predictions, pos_label=SPAM))
        # f1_score: 2 tp / (2 tp + fp + fn)
        f1_scores.append(f1_score(y_test, predictions, pos_label=SPAM))

        print("Confusion matrix for Fold No [ %d ]" % fold_no)
        print(conf_matrix)
        print("-------------------------------------")
        fold_no += 1
   
    elapsed_time = int(time() * 1000)  - start_time
    
    # Drawing final Confusion matrix
    print("--------------------------------------------------------------")
    print('Final Confusion Matrix for : ' + classifier_desc)
    print(conf_matrix)
    draw_confusion_matrix(conf_matrix, classifier_desc)
    print("--------------------------------------------------------------")
   
    # Return mean evaluation metrics
    return {'classifier_desc': classifier_desc,
            'accuracy'       : statistics.mean(accuracy_scores),
            'precision'      : statistics.mean(precision_scores),
            'recall'         : statistics.mean(recall_scores),
            'f1_score'       : statistics.mean(f1_scores),
            'elapsed_time'   : elapsed_time}


def print_dataset_info(content_df):
  print("\n****************  Dataset Info ***************")
  print("Content info by class :")
  print("-----------------------")
  print(content_df.groupby(['class']).count())
  
  show_term_frequency(content_df, HAM)
  show_term_frequency(content_df, SPAM)
 
# Load dataset
content_df = load_dataset()

# Print dataset info
print_dataset_info(content_df)

# Evaluation Dataframe
evaluation_df= DataFrame(columns=('classifier_desc', 'accuracy', 'f1_score','precision', 'recall', 'elapsed_time'))

# First pipeline
first_pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
evaluation_df = evaluation_df.append(train_and_predict(content_df, first_pipeline, "MultinomialNB"), ignore_index=True)

# Second pipeline
second_pipeline = make_pipeline(TfidfVectorizer(), LinearSVC())        
evaluation_df = evaluation_df.append(train_and_predict(content_df, second_pipeline, "LinearSVC"), ignore_index=True)

# Third pipeline
third_pipeline = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
evaluation_df = evaluation_df.append(train_and_predict(content_df, third_pipeline, "RandomForest"), ignore_index=True)

# showing evaluation result report
show_evaluation_report(evaluation_df)
