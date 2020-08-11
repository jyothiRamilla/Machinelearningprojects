# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:23:11 2020

@author: Lenovo
"""

###Data Exploration

##Convert character date to Date

from datetime import datetime
char_date = 'Apr 1 2015 1:20 PM' #creating example character date
date_obj = datetime.strptime(char_date, '%b %d %Y %I:%M %p')
print(date_obj)


##Transposing a dataframe by a variable

import pandas as pd
df=pd.read_excel("E:/transpose.xlsx", "Sheet1") # Load Data sheet of excel file EMP 
print (df)

result= df.pivot(index= 'ID', columns='Product', values='Sales') 
result 

##Sort Dataframe

#Sorting Dataframe 
df=pd.read_excel("E:/transpose.xlsx", "Sheet1") 

#Add by variable name(s) to sort print 
df.sort(['Product','Sales'], ascending=[True, False])


""" Let's look at the some of the visualization to understand below behavior of variable(s) .

The distribution of age
Relation between age and sales; and
If sales are normally distributed or not?

"""

##Histogram

import matplotlib.pyplot as plt 
import pandas as pd
df=pd.read_excel("E:/First.xlsx", "Sheet1")
#Plots in matplotlib reside within a figure object, use plt.figure to create new figure 
fig=plt.figure()
#Create one or more subplots using add_subplot, because you can't create blank figure 
ax = fig.add_subplot(1,1,1)
#Variable 
ax.hist(df['Age'],bins = 5)
#Labels and Title 
plt.title('Age distribution') 
plt.xlabel('Age') 
plt.ylabel('#Employee') 
plt.show() 

##Scatterplot

#Plots in matplotlib reside within a figure object, use plt.figure to create new figure 
fig=plt.figure()
#Create one or more subplots using add_subplot, because you can't create blank figure 
ax = fig.add_subplot(1,1,1)
#Variable 
ax.scatter(df['Age'],df['Sales'])
#Labels and Title 
plt.title('Sales and Age distribution') 
plt.xlabel('Age') 
plt.ylabel('Sales') 
plt.show() 

##Box-plot

import seaborn as sns 
sns.boxplot(df['Age']) 
sns.despine() 

##Frequency Tables can be used to understand the distribution of a categorical variable or n categorical variables using frequency tables.

import pandas as pd
df=pd.read_excel("E:/First.xlsx", "Sheet1") 
print(df)
test= df.groupby(['Gender','BMI']) 
test.size()


#Create Sample dataframe
import numpy as np 
import pandas as pd 
from random import sample
# create random index 
rindex = np.array(sample(range(len(df)), 5))
# get 5 random rows from df 
dfr = df.ix[rindex] 
print(dfr)


##How to remove duplicate values of a variable

#Remove Duplicate Values based on values of variables "Gender" and "BMI"
rem_dup=df.drop_duplicates(['Gender', 'BMI']) 
print (rem_dup)


##Grouping variables in python

test= df.groupby(['Gender']) 
test.describe() 


##To recognise missing values and outliers

df.isnull()


#Example to impute missing values in Age by the mean 
import numpy as np 

#Using numpy mean function to calculate the mean value 
meanAge = np.mean(df.Age)     

#replacing missing values in the DataFrame with the mean value
df.Age = df.Age.fillna(meanAge) 

#Here df1 and df2 are two dataframes
df_new = pd.merge(df1, df2, how = 'inner', left_index = True, right_index = True) 
# merges df1 and df2 on index 
# By changing how = 'outer', you can do outer join. 
# Similarly how = 'left' will do a left join 
# You can also specify the columns to join instead of indexes, which are used by default.






