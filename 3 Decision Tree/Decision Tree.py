#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
get_ipython().system('{sys.executable} -m pip install graphviz')


# #Graphviz是一个开源的绘图工具，可以根据指定的图形描述语言生成各种类型的图形，如流程图、组织结构图等。

# In[4]:


get_ipython().system('{sys.executable} -m pip install pydot')


# #Pydot允许开发人员使用Python代码创建和编辑图形描述，然后使用Graphviz将其转换为实际的图形。它提供了一组API，用于创建节点、边和子图，并支持各种图形布局算法，例如层次布局、圆形布局等。
# 
# #使用Pydot，开发人员可以在Python中自动化生成和操作图形，从而在数据可视化、网络拓扑分析、组织结构图等领域中发挥重要作用。它也是许多机器学习和数据科学库的重要组成部分，用于可视化模型结构和流程。

# In[5]:



from sklearn import tree
from sklearn.datasets import load_iris
import pydot
import numpy as np
import random
from numpy.random import RandomState
from scipy import stats
import math
import graphviz
import pandas as pd


# In[6]:


#read files
SDSS_data = pd.read_csv(r'skysurvey/training_data.csv',
                 names = [ 'ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'run', 'rerun', 'camcol', 'field', 'specobjid', 
                          'redshift', 'plate', 'mjd', 'fiberid'])

SDSS_class = pd.read_csv(r'skysurvey/training_class.csv',
                 names = ['target'])


# In[7]:


#print(SDSS_class)


# #### Task 1: Given the dataset we provided to you, build a decision tree using the parameter min sample leaf = 0.01. Such a parameter value specifies that the training data per leaf is 1% of all training data which allows us to get statistically significant results. Set also random state = RandomState(2018), which makes the algorithm deterministic. All other parameters should have their default values. Include the decision tree you built in your submission (stored in a pdf file).

# In[8]:


#build decision tree with the given parameter
clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=0.01, random_state = RandomState(2018))
clf = clf.fit(SDSS_data, SDSS_class)
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=SDSS_data.columns,  
                         class_names=['star', 'galaxy', 'quasar'],  
                         filled=True, rounded=True,  
                         special_characters=True)  

graphv = graphviz.Source(dot_data) 
#print(graphv)


# In[9]:


#plot the decision tree to a pdf
get_ipython().system('pip install --upgrade pyparsing')
from six import StringIO
import pydot

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=SDSS_data.columns,  
                         class_names=['star', 'galaxy', 'quasar'],  
                         filled=True, rounded=True,  
                         special_characters=True  
                    )


# In[10]:


get_ipython().system('{sys.executable} -m pip install pydotplus')
import pydotplus


# In[11]:


import os
os.environ["PATH"] += os.pathsep + '/usr/local/bin'


# In[12]:


from IPython.display import Image
from pydotplus import graph_from_dot_data

graph = graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

with open("Sky Survey.pdf", "wb") as pdf:
    pdf.write(graph.create_pdf())


# #### Task 2: compute the generalization error of the decision tree you built. To this end, you might use the array clf.tree.children left where clf.tree.children left[i] = −1 if i is a leaf while clf is the tree you built with DecisionTreeClassifier in sci-kit learn.

# In[13]:


leaves_num = (clf.tree_.children_left == -1).sum()
print(leaves_num)


# In[14]:


error_train = (SDSS_class['target'] != clf.predict(SDSS_data)).sum()
print(error_train)


# In[15]:


error_general = error_train + 0.5*leaves_num
print('The generalization error of the decision tree is {}'.format(error_general))


# #### Task3: the decision tree you built in the first part of the question might not be ideal for our task. You should try to change the input parameters of DecisionTreeClassifier, so as to build a decision tree with minimum gen- eralization error. 
# Here we consider the parameter max depth. Determine the best value for max depth so as to minimize the generalization error. You should maintain min sample leaf = 0.01 so as to make sure to ob- tain results that are statistically significant. Do not change random state either. Specify in the answer to this question, which values for max depth you considered and how you expect that a given value affects the generalization error. It should be clear from your answer that you understood what is the role of max depth and how it might affect the generalization error. Include the decision tree you built in the report.

# In[16]:


#test difference max_depth
clf1 = tree.DecisionTreeClassifier(criterion='gini',max_depth = 2, min_samples_leaf=0.01, random_state = RandomState(2018))
clf1 = clf1.fit(SDSS_data, SDSS_class)
leaves_num1 = (clf1.tree_.children_left == -1).sum()
error_train1 = (SDSS_class['target'] != clf1.predict(SDSS_data)).sum()
error_general1 = error_train1 + 0.5*leaves_num1
print('The generalization error of the decision tree is {}'.format(error_general1))


# In[17]:


clf2 = tree.DecisionTreeClassifier(criterion='gini',max_depth = 10, min_samples_leaf=0.01, random_state = RandomState(2018))
clf2 = clf2.fit(SDSS_data, SDSS_class)
leaves_num2 = (clf2.tree_.children_left == -1).sum()
error_train2 = (SDSS_class['target'] != clf2.predict(SDSS_data)).sum()
error_general2 = error_train2 + 0.5*leaves_num2
print('The generalization error of the decision tree is {}'.format(error_general2))


# In[18]:


#define a function to test more max_depth
def genError (max_depth):
    clf2 = tree.DecisionTreeClassifier(criterion='gini',max_depth = max_depth, min_samples_leaf=0.01, random_state = RandomState(2018))
    clf2 = clf2.fit(SDSS_data, SDSS_class)
    leaves_num2 = (clf2.tree_.children_left == -1).sum()
    error_train2 = (SDSS_class['target'] != clf2.predict(SDSS_data)).sum()
    error_general2 = error_train2 + 0.5*leaves_num2
    
    print('The generalization error of the decision tree with max_depth {}, is {}'.format(max_depth, error_general2))
    #return clf2, error_general2
    


# In[19]:


#what is the depth of the decision tree?
depth = clf2.tree_.max_depth
print(depth)


# In[20]:


for i in [1,2,5,10,11,20,None]:
    genError(i)


# In[21]:


for i in [2,3,4,5]:
    genError(i)


# #### Answer:
# 
# The `max_depth` parameter in `DecisionTreeClassifier` controls the maximum depth of the decision tree. It limits the complexity and generalization ability of the model, preventing overfitting (high values) or underfitting (low values). 
# 
# The generalization error is affected by a combination of two factors: the training error and the number of leaf nodes.
# 
# In general, when the depth increasing, the number of training errors should goes down because there are more attributes splitting branches to fit the date. On the contrary, The number of leaf nodes should increase.
# 

# I first queried the depth of the decision tree. Found the current depth to be 10.
# 
# So I entered 1, 2, 5, 10, 11, 20 and None as max_depth parameter.
# According to my judgment, when max_depth = 1, the decision tree should only have two layers, a root and a layer of leaf nodes, and the training error at this time should be the largest. The output matches my guess.
# 
# At the same time, I suppose that when the value of max_depth is larger than the depth of the decision tree itself, the value of gen error will remain unchanged. Because the decision tree will not continue to split the nodes that don't meet the splitting requirements. The gen error remains the same when we change the max_depth among 11, 20 and None.
# 
# With the first test, I found that the gen error is smaller than several others when the max_depth is 2. So I continued to test the gen error for max_depth of 2, 3, 4, and 5. Then found that 2 is still the smallest. This may mean that one half of the number of extra leaf nodes generated when depth increasing exceeds the number of reduced training errors.

# #### Task4: compare the decision trees you built in point 1 and the best one you obtained in point 3. Which one would you recommend to use to classify sky objects? Explain your answer.

# In[22]:


#plot the optimal tree clf1 with max_depth=2


dot_data = StringIO()
tree.export_graphviz(clf1, out_file=dot_data, feature_names=SDSS_data.columns,  
                         class_names=['star', 'galaxy', 'quasar'],  
                         filled=True, rounded=True,  
                         special_characters=True  
                    )

graph = graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

with open("Sky Survey depth2.pdf", "wb") as pdf:
    pdf.write(graph.create_pdf())


# #### Answer:
# 
# I would recommend the decision tree 'clf1' with max_depth = 2. Because the generalization error of 'clf1' is 115 and that of the decision tree 'clf' in point1 is 127. The new tree has smaller generalization error which means that we should post-pruning the original decision tree.

# #### Task5: Consider the decision tree you considered to be best in the pre- vious point. Predict the class value of an object of your choice. Which feature is most relevant when classifying sky objects?

# In[23]:


print("The 1st instance of the training sample {} is predicted to be {}.".format(
    SDSS_data.loc[0].to_dict(), ['star', 'galaxy', 'quasar'][clf1.predict([SDSS_data.loc[0]])[0]]))


# In[24]:


print(SDSS_data.columns[clf1.feature_importances_.argmax()], "is considered to be the most relevant.")


# `feature_importances_.argmax()` is used to identify the most important feature by finding the index of the feature with the highest importance score in the `DecisionTreeClassifier` model.
# 
# redshift is considered to be the most relevant attributes in the decision tree.
# 
# When redshift <= 0.002 ➡️ 'star'
# 
# When 0.002< redshift <= 0.218 ➡️ 'galaxy'
# 
# When redshift > 0.218 ➡️ 'quasar'

# #### Task6: do you think that the best decision tree you built could be pruned so as to improve the generalization error? Explain your answer (you are supposed to answer this question by only looking at the tree, no implementation is required).

# #### Answer:
# By looking at the pdf 'Sky Survey depth2', I find the leaf nodes' classes of the left root are both 'star'. Which means that the training error did not be decreased by the splitting, while the number of leaves increased 1. This part can be pruned.

# #### Task7: the library we recommend (sci-kit learn) does not support post- pruning, yet. However this could be implemented by using the variables ofthetree objectcomputedbytheDecisionTreeClassifierinsci-kitlearn. 
# See 1 to see some examples. In particular, clf.tree .children left[i] spec- ifies the index of the left children of i, clf.tree .children right[i] specifies the index of the right children of i, while clf.tree .value[i] specifies the class distribution of i. Implement a post-pruning strategy (among the ones we considered in our course) and run it on the best decision tree so far. Does this improve the generalization error? In case you cannot modify your instance of the DecisionTreeClassifier, you can use another data structure to store the pruned tree.
# 

# In[25]:


#collect the training errors, leaf node numbers and depth of the tree
def trainError (index, children_left, children_right, values):
    if children_left[index] == -1:
        return values[index].sum() - values[index].max()
    return trainError(children_left[index], children_left, children_right, values) + trainError(children_right[index], children_left, children_right, values)

def leafNum (index, children_left, children_right):
    if children_left[index] == -1:
        return 1
    return leafNum(children_left[index], children_left, children_right) + leafNum(children_right[index], children_left, children_right)

def search(index, children_left, children_right):
    if children_left[index] == -1:
        return 0
    return max (search(children_left[index], children_left, children_right),
               search(children_right[index], children_left, children_right)) + 1


# In[26]:


#get the index of each layer in the decision tree
nodes_num = len(clf1.tree_.value)
subtree_left = list(clf1.tree_.children_left)
subtree_right = list(clf1.tree_.children_right)
#subtree_left

tree_pruned = {}

for i in range(nodes_num):
    layer = search(i, subtree_left, subtree_right)
    if layer not in tree_pruned:
        tree_pruned[layer] = []
    tree_pruned[layer].append(i)
    
print(tree_pruned)


# In[27]:


#post pruning if generalization error can decrease

tree_depth = max(tree_pruned.keys())

for layer in range (1, tree_depth):
    for node in tree_pruned[layer]:
        increase_gen_error = trainError(node, subtree_left, subtree_right, clf1.tree_.value) - (
        clf1.tree_.value[node].sum() - clf1.tree_.value[node].max()) + 0.5*(leafNum(node, subtree_left, subtree_right)-1) 
        if increase_gen_error > 0.49:
            print('Node {} is pruned. Generalization error has gone down by {}'.format(node, increase_gen_error))
            subtree_left[node] = subtree_right[node] = -1


# In[ ]:




