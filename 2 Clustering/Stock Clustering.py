#!/usr/bin/env python
# coding: utf-8

# # K-Means Clustering Algorithm

# TASK：In this assignment, we cluster stocks in the stock market by using the k-means algorithm. In particular, you are provided with a dataset (available on the moodle website) which specifies for each of 30 stocks the percentage change in price of that stock in each given week, for a total of 25 weeks. In our dataset, some stocks might deal with technology, some other with oil, etc. We will try to group together stocks with similar price trends in the stock market. In other words, in a same cluster we would like to have stocks whose price changes by similar amounts every week. This can be used for coming up with successful investment policies. We will see that stocks related to the same market (e.g. technology) have often “similar” price trends. For this assignment, we recommend k = 8.

# ## This solution employ scikit-learn library.
# Scikit-Learn, also known as sklearn is a python library to implement machine learning models and statistical modelling. Through scikit-learn, we can implement various machine learning models for regression, classification, clustering, and statistical tools for analyzing these models. (from Google)

# In[1]:


from sklearn.cluster import KMeans


# learn more about K-Means parameters
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

# ### Q1: You should run the k-means algorithm on the stock data, while using init=’random’ and the default values for the other parameters. Compute the sum of squared errors (SSE) for the clustering you obtained and include it in your report.

# In[2]:


import pandas as pd

# read file
stockdata = pd.read_csv('clustering_data.csv', index_col=0)


# In[3]:


#print(stockdata)


# In[12]:


sdKmeans = KMeans(init='random', n_clusters=8).fit(stockdata.values)
SSE1 = sdKmeans.inertia_


# ### Answer to Q1: Because of the random selection of initial centroids, the SSE result changes everytime when the function runs.

# In[13]:


print('The SSE for the clustering of default parameters is {}.'.format(SSE1))


# ### Q2:You should then try to decrease the SSE as much as possible (while keep- ing k = 8) by changing some of the parameters accordingly. To this end, select two parameters that you think should impact the results the most. For each parameter explain : 
# a) how you expect that changing that parameter would affect the results (increasing its value means better or worse results?)
# b) whether increasing or decreasing the value of the parameter should always improve the results or not necessarily.

# In[20]:


# adopt a loop to test the improvement possibility of changing init parameter to k-means++
df = 0
km1 = 0

for i in range(200):
    # calculate the default set again
    sdKmeans_df = KMeans(init='random', n_clusters=8).fit(stockdata.values)
    SSE_df = sdKmeans_df.inertia_

    # try max_iter
    sdKmeans2 = KMeans(init='k-means++', n_clusters=8).fit(stockdata.values)
    SSE2 = sdKmeans2.inertia_
    
    if SSE_df <= SSE2:
        df += 1
    else:
        km1 += 1


# In[22]:


# in 200 runs, lets see the best clustering solution distributions among
# default parameter and k-means++
print('Default set wins {} times.'.format(df))
print('K-means++ set wins {} times.'.format(km1))


# In[30]:


# adopt a loop to test the improvement possibility of changing max_iter parameter
dfnew = 0
ni1 = 0
ni2 = 0

for i in range(100):
    # calculate the default set again
    sdKmeans_df2 = KMeans(init='random', n_clusters=8).fit(stockdata.values)
    SSE_df2 = sdKmeans_df2.inertia_

    # try n_init
    sdKmeans4 = KMeans(init='random', n_clusters=8, n_init = 100).fit(stockdata.values)
    SSE4 = sdKmeans4.inertia_
    sdKmeans5 = KMeans(init='random', n_clusters=8, n_init = 1000).fit(stockdata.values)
    SSE5 = sdKmeans5.inertia_

    if SSE_df2 <= SSE4 and SSE_df2 <= SSE5:
        dfnew += 1
    elif SSE4 <= SSE_df2 and SSE4 <= SSE5:
        ni1 += 1
    elif SSE5 <= SSE_df2 and SSE5 <= SSE4:
        ni2 += 1


# In[31]:


# in 100 runs, lets see the best clustering solution distributions among
# default parameter
# n_init = 100
# n_init = 1000
print('Default set wins {} times.'.format(dfnew))
print('Lower n_init set wins {} times.'.format(ni1))
print('Higher n_init set wins {} times.'.format(ni2))


# ### Answer to Q2: 

# I chose two parameters 'init' and 'n_init', which I think can improve the performance of clustering. 
# 
# For the parameter 'init', I chose 'k-means++'. It refers to the slection of initial cluster centroids using sampling based on an empirical probability distribution of the points' contribution to the overall inertia. By choosing 'k-means++' will end up with better results in most times. However, this parameter is not necessarily improving the clustering performance as the random selection of initial centroids may be better than applying k-means++ algorithm sometimes.
# 
# For the parameter 'n-init', it refers the number pf times the k-means algorithm is run with different centroid seeds. As can be seen from the loop results, the larger n_init is, the higher the likelihood of obtaining better clustering results. But increasing n_init means that more calculations have to be performed, which increases the cost of the algorithm and reduces its efficiency.

# In[ ]:





# ### Q3:Then look at the clustering you obtained and try to label each cluster with a topic. For example: cluster of technology stocks, oil stocks, etc. Don’t expect your clustering to be perfect. In particular, you might have different kinds of stocks in a given cluster, while you might not be able to label all clusters. We expect that you should be able to label at least three clusters with a topic. It is fine to describe a cluster as a technology cluster if most of the stocks deal with technology, for example. Explain your answers.

# In[32]:


# choose a better clustering

SSE_df,SSE2,SSE_df2,SSE4,SSE5


# In[33]:


# SSE5 is the lowest, therefore
pd.DataFrame(stockdata.index, index=sdKmeans5.labels_).sort_index()


# ### Answer to Q3: According to the clustering above, I can label the Cluster 0，1，5，6:

# Cluster0: Energy stock

# Cluster1: Consumer Goods stock

# Cluster5: Manufacturing and Industrial stock

# Cluster6: Financial and Technology stock

# In[ ]:




