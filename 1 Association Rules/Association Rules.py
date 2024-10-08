#!/usr/bin/env python
# coding: utf-8

# # Association Rules Algorithm

# TASK:You are provided with a dataset which contains some data resulting from Mammography tests.
# Description of dataset. Mammography is the most effective method for breast cancer screening available today. However, the low positive predictive value of breast biopsy resulting from mammogram interpretation leads to ap- proximately 70% unnecessary biopsies with benign outcomes. To reduce the high number of unnecessary breast biopsies, several computer-aided diagnosis (CAD) systems have been proposed in the last years.These systems help physi- cians in their decision to perform a breast biopsy on a suspicious lesion seen in a mammogram or to perform a short term follow-up examination instead. The dataset we provided can be used to predict the severity (benign or malignant) of a mammographic mass lesion from the attributes of the mass lesion (size, shape, etc.) and the patient’s age. Additionally, the dataset contains for each patient the BI-RADS assessment, which is a score ranging from 0 (definitely benign) to 6 (highly suggestive of malignancy) given by the radiologist upon checking the results of the test. The ground truth is also provided (the severity field), which specifies whether the corresponding mass lesion is benign (0) or malign (1). In the dataset, there are 516 benign and 445 malignant masses that have been identified on full field digital mammograms collected at the Institute of Radiology of the University Erlangen-Nuremberg (Germany) between 2003 and 2006.

# ## This solution employ mlxtend library.
# Mlxtend (machine learning extensions) is a Python library of useful tools for the day-to-day data science tasks. (from Github)

# In[14]:


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

import math
import pandas as pd


# In[33]:


# the dataset needs preprocessing to one-hot encoding  
# and remove any NaN values to use apriori class
dataSet = pd.read_csv(r"mammographic_masses.csv",delimiter=",",header=0)
dataList = dataSet.values.tolist()

for i in range(len(dataList)):
    j=0
    while (True):
        if (type (dataList[i][j]) == float and math.isnan(dataList[i][j])):
            del dataList[i][j]
            j -= 1
        j += 1
        if (j>len(dataList[i])-1):
            break

#dataList


# ### Q1: Report 3 rules with support at least 0.2 and confidence at least 0.9. Specify for each of them the support and the confidence.

# In[34]:


# prepare clear demonstration for later output
for i in range (len(dataList)):
    for j in range (len(dataList[i])):
        dataList[i][j] = dataSet.columns[j] + '= ' + str(dataList[i][j])
dataList


# In[35]:


# create a Boolean list for apriori algorithm
te = TransactionEncoder()
teArray = te.fit(dataList).transform(dataList)
dataF = pd.DataFrame(teArray, columns = te.columns_)


# In[36]:


# calculate frequent itemsets of 0.2 support and 0.9 confidence
frequentItemSets = apriori (dataF, min_support=0.2, use_colnames=True)
resultAR = association_rules (frequentItemSets, metric='confidence', min_threshold=0.9)
resultAR [["antecedents","consequents","support","confidence"]]


# ### Anwser to Q1:

# From the sheet above, we can see the first three rules are:

# Shape=4 -> Density=3 has 0.37 support and 0.9 confidence.

# Margin=1, Density=3 -> BI-RADS=4 has 0.26 support and 0.9 confidence.

# Marigin=1, Severity=0 -> BI-RADS=4 has 0.3 support and  0.91 confidence.

# In[ ]:





# ### Q2: Report one or two rules with the specified requirements that you think might help us predicting whether a given instance is benign or malign. You should not report rules with the attribute BI-RADS for this question. Which insights did you get from those rules? (e.g. the margin of the lesion can help us determining whether a lesion is benign or malign).

# Remember that only rules with support at least 0.1, (i.e. their frequency is at least 10%) are relevant for us. Rules with lower support are usually not informative, as there is no much evidence they are true or not. In our exercise we consider relevant any rule with confidence at least 0.9 (i.e. they are true 90% of times).

# In[40]:


frequentItemSets2 = apriori (dataF, min_support=0.1, use_colnames=True)
resultAR2 = association_rules (frequentItemSets2, metric='confidence', min_threshold=0.9)
resultAR2 [["antecedents","consequents","support","confidence"]]


# In[43]:


# eliminates the rows according to the requirements
# consequents should be Severity
# no BI-RADS involved

rightAn = []

for i, row in resultAR2.iterrows():
    if ('Severity= 1' in row['consequents']) or ('Severity= 0' in row['consequents']):
        if ('BI-RADS= 0' not in row['antecedents']) and ('BI-RADS= 1' not in row['antecedents']) and ('BI-RADS= 2' not in row['antecedents']) and ('BI-RADS= 3' not in row['antecedents']) and ('BI-RADS= 4' not in row['antecedents']) and ('BI-RADS= 5' not in row['antecedents']) and ('BI-RADS= 6' not in row['antecedents']):
            rightAn.append(i)
print(rightAn)


# ### Anwser to Q2:

# From the list, we find 12th and 28th rows meet the requirements.

# 12	(Margin= 1, Shape= 2) -> (Severity= 0)	support=0.136316	confidence=0.903448

# 28	(Margin= 1, Density= 3, Shape= 1) -> (Severity= 0)	support=0.143600	confidence=0.901961

# #### Insights from the rules:

# If a lesion has an oval shape and a circumscribed margin, it is more likely to be benign. Similarly, if a lesion has a round shape, a circumscribed margin, and low density, it is also more likely to be benign.

# In[ ]:





# ### Q3: As discussed above, the BI-RADS assessment is not always accurate and it might lead to unnecessary breast biopsy. Provide one or two rules that might give some evidence that the BI-RADS assessment is not always accurate. Explain your answer.

# Firsly, assume BI-RADS above 3 is suppose to reveal malignancy. Let's check how offen high BI-RADS relating to benign. To understand why unnecessary breast biopsy exist, we need to check how offen low BI-RAD relating to malignancy as well.

# In[51]:


# We use resultAR2 again here

rightAn2 = []
for i, row in resultAR2.iterrows():
    if ('Severity= 0' in row['consequents']):
        if ('BI-RADS= 4' in row['antecedents']) or ('BI-RADS= 5' in row['antecedents']) or ('BI-RADS= 6' in row['antecedents']):
            rightAn2.append(i)

resultAR2.loc[rightAn2, ["antecedents","consequents","support","confidence"]]


# In[52]:


rightAn3 = []
for i, row in resultAR2.iterrows():
    if ('Severity= 1' in row['consequents']):
        if ('BI-RADS= 0' in row['antecedents']) or ('BI-RADS= 1' in row['antecedents']) or ('BI-RADS= 2' in row['antecedents']):
            rightAn3.append(i)

resultAR2.loc[rightAn3, ["antecedents","consequents","support","confidence"]]


# ### Anwser to Q3:

# From the result above, we can find BI-RADS=4 appearing a lot in antecedents when the consequents is Severity=0. The listed association rules all have support over 0.1 and confidence over 0.9. Based on the results, we can assume that when BI-RADS = 4, it often leads to UNNECESSARY biopsies with benign outcomes. 
# 

# However, we can't dismiss the instructive nature of BI-RADS because through data mining, we can see that low BI-RADS value and Severity=1 are never strongly correlated. So this indicator hardly leads to missed diagnosis of malignant chest tumors, for example.
# We need to further improve the association rules by data mining on the current basis to reduce unnecessary biopsies.

# In[ ]:





# ### Q4: Write a script in Python to find the confidence and support of the following rule: Age=35 ⇒ Severity=0. Report its support and confidence. Do you think this rule tells us something valuable or that we should ignore it as there is not enough evidence to support this rule?

# In[53]:


# the support and confidence are unknow, therefore we lower the threshold
frequentItemSets4 = apriori (dataF, min_support=0.0001, use_colnames=True)
resultAR4 = association_rules (frequentItemSets4, metric='confidence', min_threshold=0)
resultAR4 [["antecedents","consequents","support","confidence"]]


# In[55]:


rightAn4 = []

for i, row in resultAR4.iterrows():
    if ('Severity= 0' in row['consequents']) and (len(row['consequents'])==1):
        if ('Age= 35' in row['antecedents']) and (len(row['antecedents'])==1):
            rightAn4.append(i)
            break

resultAR4.loc[rightAn4, ["antecedents","consequents","support","confidence"]]


# ### Anwser to Q4:

# The support of Age=35 -> Severity=0 is 0.012, while the confidence is 0.92. Although the confidence is high in this rule, the support not high enough, which means that not enough data category under Age=35 to draw a solid conclusion. 

# In[ ]:





# ### Q5: Provide at least one rule of that kind with support at least 0.1 and confidence at least 0.9.

# The attribute “Age” is ordinal which makes the rule mining approach not ideal. In particular, one would like to obtain rules of the kind
# Age≥ n,A1 = a1,...,Ak = ak − > Severity=’1’
# (where n is an integer), as the age is an important factor in determining whether a given instance is malign or benign. However, this issue can be circumvented in our case by modifying the input file (the ’csv’ file) accordingly. Be careful on how you handle the missing values (i.e. those with a ’?’).

# In[77]:


# processing the age attribute to nominal
# '?' is replaced by NaN, as replacing '?' with certain numbers may arise wrong association rules

newData = pd.read_csv(r"mammographic_masses.csv",delimiter=",",header=0, na_values='?')
ages = list(age for age in newData['Age'].unique() if not math.isnan(age))

# we use 40s,50s,60s etc. to category the ordinal ages to nominal
age10 = [age for age in range (100,0,-10)]
#ages,age10


# In[87]:


ageGroup = []
for i, row in newData.iterrows():
    ageGroup_row = []
    for var in row.index:
        # handle NaN
        if math.isnan(row[var]):
            continue
        if var == 'Age':
            for i in age10:
                if row[var] >= i:
                    ageGroup_row.append(var + '= ' + str(i) + 's')
                    break
        else:
            ageGroup_row.append(var + '= ' +str(row[var]))
    ageGroup.append(ageGroup_row)
    
ageGroup


# In[88]:


te2 = TransactionEncoder()
teArray2 = te2.fit(ageGroup).transform(ageGroup)

dataF2 = pd.DataFrame(teArray2, columns = te2.columns_)


# In[89]:


frequentIS = apriori (dataF2, min_support=0.1, use_colnames=True)
newAR = association_rules (frequentIS, metric='confidence', min_threshold=0.9)
newAR [["antecedents","consequents","support","confidence"]]


# ### Anwser to Q4:

# 3	(Severity= 1.0, Age= 60s)	(Density= 3.0)	0.131113	0.926471

# The rule related to age attributes is as follow:
# Severity=1.0, Age=60s -> Density=3.0 has support of 0.13 and confidence of 0.93.

# In[ ]:




