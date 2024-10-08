#!/usr/bin/env python
# coding: utf-8

# In[17]:


#read dataset-news
from os import listdir
from os.path import isfile, join

title = []
articles = []
topics = ['auto', 'baseball', 'electronics', 'hockey', 'ibm-hw', 'moto', 'pol-guns', 'mac-hw']
path = r'dataset-news'


# In[18]:


for i in listdir(path):
    for j in topics:
        if i.startswith(j): #是否以topic中的单词开头
            title.append(j)
            file = open(join(path,i), 'r') #join用于拼接文件路径（指向文件名）
            articles.append(file.read())
            file.close()
#len(title)
#len(articles)


# #### 1. turn each document into a vector in the Euclidean space. Each dimension of the vector corresponds to a word, with its value specifying the number of occurrences of the corresponding word in the corresponding document. This can be done by using CountVectorizer in scikit-learn (see tutorial). It helps to remove so called “stopwords”, which are words that occur often in the English language such as articles (e.g. ’the’,’a’,), adjectives (e.g. ’my’,’yours) etc. Stopwords usually do not carry much information about the topic of a document and should be removed.
# 

# In[19]:


from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer(stop_words = 'english') #去除常见英文单词
text_vector = count_vector.fit_transform(articles)


# In[20]:


text_vector.shape


# #### 2. train a naive Bayes classifier on the collection of documents. In Python, we are going to consider the multinomial naive Bayes classifier and the Gaussian naive Bayes classifier. See tutorial.

# In[28]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(text_vector.toarray(),title)

y_pred = gnb.predict(text_vector.toarray())
print ("Number of mislabeled points in Gaussian Naive Bayes algorithm out of a total %d points : %d" % (text_vector.shape[0],(title != y_pred).sum()))


# In[29]:


from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB(alpha=1.0)
mnb.fit(text_vector, title)

y_pred2 = mnb.predict(text_vector)
print ("Number of mislabeled points in Multinominal Naive Bayes algorithm out of a total %d points : %d" % (text_vector.shape[0],(target != y_pred2).sum()))


# #### 3. perform a 10-fold cross validation to determine which classifier performs best. More informations on how to do that are provided in the tutorial.
# 

# #### Task a: perform a 10-fold cross validation and report the average accuracy (i.e. average number of documents classified correctly) for both the multinomial and Gaussian naive Bayes classifier.

# In[31]:


import numpy as np
from sklearn.model_selection import KFold

kf = KFold(n_splits=10)


# In[32]:


#Gaussian Naive Bayes classifier

counter = 0
accuracy = []

for train, test in kf.split(text_vector):
    train_split1 = text_vector.toarray()[train]
    train_split2 = np.array(title)[train]
    test_split1 = text_vector.toarray()[test]
    test_split2 = np.array(title)[test]
    counter += 1
    
    gnb = GaussianNB()
    gnb.fit(train_split1, train_split2)
    
    accuracy.append(gnb.score(test_split1, test_split2))
    
print ('Gaussian NB classifier has the average accuracy of {}'.format(sum(accuracy)/len(accuracy)))


# In[36]:


#Multinominal NB classifier

counter1 = 0
accuracy1 = []

for train, test in kf.split(text_vector):
    train_split1 = text_vector.toarray()[train]
    train_split2 = np.array(title)[train]
    test_split1 = text_vector.toarray()[test]
    test_split2 = np.array(title)[test]
    
    counter1 += 1
    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(train_split1, train_split2)
    
    accuracy1.append(mnb.score(test_split1, test_split2))
    
print ('Multinominal NB classifier has the average accuracy of {}'.format(sum(accuracy1)/len(accuracy1)))


# #### Task b: consider the best classifier according to your evaluation. How does such a classifier compare with a random classifier which classifies the documents randomly? (i.e. it assigns to each document one of the eight topics, chosen randomly). To answer this question, compute the accuracy of the random classifier (i.e. the probability that a document is classified correctly) and include it in your Jupyter notebook.

# In[39]:


#calculate accuracy of random classifier

import random
random_vector = []
for i in range(len(title)):
    random_vector.append(random.choice(topics))
    
#random_vector


# In[40]:


from sklearn.metrics import accuracy_score
print("Random classifier has an accuracy score of {}.".format(accuracy_score(title, random_vector)))


# The accuracy score of a random classifier is far below the Gaussian NB classifier and the Multinominal NB classifier.

# #### Task c: report the accuracy of the best classifier when the “stopwords” are not removed. Does the accuracy improve or worsen? Try to explain why the accuracy improves or worsens.
# 

# In[46]:


count_vector_none = CountVectorizer(stop_words = None) #不去除常见英文单词
text_vector_none = count_vector_none.fit_transform(articles)


# In[47]:


kf = KFold(n_splits=10)


# In[48]:


#Multinominal NB classifier
accuracy3 = []
counter3 = 0

for train, test in kf.split(text_vector_none):
    train_split1 = text_vector_none.toarray()[train]
    train_split2 = np.array(title)[train]
    test_split1 = text_vector_none.toarray()[test]
    test_split2 = np.array(title)[test]
    
    counter3 += 1
    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(train_split1, train_split2)
    
    accuracy3.append(mnb.score(test_split1, test_split2))
    
print ('(Stopwords are not removed) Multinominal NB classifier has the average accuracy of {}'.format(sum(accuracy3)/len(accuracy3)))


# The accuracy for Multinominal NB classifier for text vectors that removed stopwords is 0.7875, which is slightly higher. 
# It is possible that the presence of stopwords can lead to overfitting of the classifier to the relationship between them and the title. This is because stopwords may not carry useful information and may increase noise in the data, which can negatively affect the performance of the classifier. Therefore, removing stopwords can help improve the accuracy and generalization ability of the classifier.

# #### Task d: Consider the following two classification tasks. In the first task, you should consider only the documents related to the following topics: “use of guns”, “hockey”,“Mac hardware”, while in the second task you should consider “Mac hardware”,“IBM hardware” and “electronics”. Perform a 10-fold cross validation for each of the two tasks and report the average accuracy. Try to explain why one task might be easier than the other one.

# #### Task1

# In[50]:


index1 = [i for i in range(len(title)) if title[i] in (["pol-guns", "hockey", "mac-hw"])]
task1_vector = text_vector.toarray()[index1]
title_task1 = np.array(title)[index1]
kf = KFold(n_splits=10)


# In[51]:


accuracy4 = []
counter4 = 0

for train, test in kf.split(task1_vector):
    train_split1 = task1_vector[train]
    train_split2 = title_task1[train]
    test_split1 = task1_vector[test]
    test_split2 = title_task1[test]
    
    counter4 += 1
    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(train_split1, train_split2)
    
    accuracy4.append(mnb.score(test_split1, test_split2))
    
print ('MNB classifier for Task1 has the average accuracy of {}'.format(sum(accuracy4)/len(accuracy4)))


# #### Task2

# In[52]:


index2 = [i for i in range(len(title)) if title[i] in (["electronics", "ibm-hw", "mac-hw"])]
task2_vector = text_vector.toarray()[index2]
title_task2 = np.array(title)[index2]
kf = KFold(n_splits=10)


# In[53]:


accuracy5 = []
counter5 = 0

for train, test in kf.split(task2_vector):
    train_split1 = task2_vector[train]
    train_split2 = title_task2[train]
    test_split1 = task2_vector[test]
    test_split2 = title_task2[test]
    
    counter5 += 1
    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(train_split1, train_split2)
    
    accuracy5.append(mnb.score(test_split1, test_split2))
    
print ('MNB classifier for Task2 has the average accuracy of {}'.format(sum(accuracy5)/len(accuracy5)))


# The accuracy for Task 1 is 0.94 and the accuracy for Task 2 is 0.76. We can find that Task 1 is easier than Task 2. 
# The decrease in accuracy without removing stopwords may be attributed to the significant variations in text among classes in task 1 ("use of guns", "hockey", "Mac hardware") compared to the relatively smaller differences in text among classes in task 2 ("Mac hardware", "IBM hardware", and "electronics").

# In text classification tasks, the performance of a classifier may be inferior for a group of articles with similar topics compared to a group of articles with diverse topics. This is because articles with similar topics often contain similar vocabulary and phrases, which results in similar word frequencies across different categories. This makes it challenging for the classifier to differentiate key features between different categories, leading to an increase in classification errors. Conversely, for a group of articles with diverse topics, the classifier can more easily identify differences between different categories, resulting in better performance.

# In[ ]:




