# Ham-Spam Email Classification With Machine Learning

Spam e-mails have been a chronic and serious problem in our business and personal life. They are very costly in terms of economic and extremely dangerous for people, computers and networks. Therefore, they should be efficently detected and prevented. All e-mail service providers and modern spam filtering software are continuously struggling to detect unwanted e-mails and mark them as spam e-mail. Managing and classifying e-mails is a big challenge.

In this comparative study, some of the popular machine learning algorithms train on diverse size of dataset and identify whether an email is spam or ham. Before using algorithms for prediction, commonly Tf-Idf method is used for feature extraction. Naïve Bayes, Support Vector Machine and Random Forest are the compared classifiers. 3 sample of dataset which has different size and distribution of e-mail type (ham/spam ratio) was used from Enron Email Dataset [30] which has more than 30000 different emails.  Thus we can observe how dataset size affects classifier performance. The performance of the classifiers was evaluated based on some metrics like accuracy, recall, precision and f1-score. Also total elapsed time for training and testing activities on classifier basis takes into consideration. 

The main goal of this study is to design and develop a content based spam-ham classification system for emails, also to compare classifiers whether which is more efficient and reliable. Python programming language with Nltk and Scikitlearn modules was used for project implementation

# Tools & Techniques
* Python (3.8)
* Spyder
* Scikit-Learn, numpy, pandas, glob, statistics
* Nltk (Stopwords, Lemmatizer)
* WordCloud, Matplotlib.pyplot  (Visualization)
* k-Fold Cross-Validation (Stratified)
* Pipelines
* Feature exctractor : TfidfVectorizer
* Classifiers : MultinomialNB, LinearSVC, RandomForestClassifier
