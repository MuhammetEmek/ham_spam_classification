# Ham-Spam Email Classification With Machine Learning

Spam e-mails have been a chronic and serious problem in our business and personal life. They are very costly in terms of economic and extremely dangerous for people, computers and networks. Therefore, they should be efficently detected and prevented. All e-mail service providers and modern spam filtering software are continuously struggling to detect unwanted e-mails and mark them as spam e-mail. Managing and classifying e-mails is a big challenge.

In this comparative study, some of the popular machine learning algorithms train on diverse size of dataset and identify whether an email is spam or ham. Before using algorithms for prediction, commonly Tf-Idf method is used for feature extraction. Naïve Bayes, Support Vector Machine and Random Forest are the compared classifiers. 3 sample of dataset which has different size and distribution of e-mail type (ham/spam ratio) was used from Enron Email Dataset [30] which has more than 30000 different emails.  Thus we can observe how dataset size affects classifier performance. The performance of the classifiers was evaluated based on some metrics like accuracy, recall, precision and f1-score. Also total elapsed time for training and testing activities on classifier basis takes into consideration. 

The main goal of this study is to design and develop a content based spam-ham classification system for emails, also to compare classifiers whether which is more efficient and reliable. Python programming language with Nltk and Scikitlearn modules was used for project implementation

# Tools & Techniques
* Python (3.8)
* Spyder IDE
* Scikit-Learn, numpy, pandas, glob, statistics, nltk
* Pipelines
* Stratified k-Fold Cross-Validation
* Visualization : WordCloud, Matplotlib.pyplot
* Feature exctractor : TfidfVectorizer
* Classifiers : MultinomialNB, LinearSVC, RandomForestClassifier

# Dataset
Enron Email Dataset which has 6 separate datasets that each contains about 6000 individual emails as described in the paper [31] was used in this study (Table-2). Each email message is in a separate text file includes numbers, alphabets and characters, also the number at the beginning of each filename is the "order of arrival". The dataset is more realistic than previous comparable benchmarks, because they maintain the temporal order of the messages in the two categories, and they emulate the varying proportion of spam and ham messages that users receive over time.

![Enron_Dataset](https://github.com/MuhammetEmek/ham_spam_classification/blob/main/enron_dataset.png)

# Evaluation
LinearSVC classifier reached the highest accuracy and f1score, and also its performance (computation time) is near the MultinomialNB. On the other hands, it wasn’t affected by the size of dataset and class type (ham-spam) distribution, so it is very stable, consistent and efficient. 

MultinomialNB classifier reached the best performance considering the computation time. Also its accuracy and f1score was close to LinearSVC results when trained and tested with whole dataset. But it was affected by the size of dataset and class type distribution. Depending the difference in ratio of ham and spam emails in the dataset, the recall value and f1-score can be change. The worst evaluation results were measured when the ratio of spam/ham was 1/3 by using Enron-1, Enron-2 and Enron-3. According to these results, this classifier isn’t consistent. 

RandomForest classifier’s performance considering the computation time was the worst. There are approximately %2000 (20 times) difference between others, for this reason it isn’t not acceptable. Other evaluation results is close to LinearSVC. Finally, LinearSVC is better than the others according to the evaluation results compared above

![Evaluation_Results](https://github.com/MuhammetEmek/ham_spam_classification/blob/main/evalution_result.PNG)
