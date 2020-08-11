# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:49:29 2020

@author: Lenovo
"""

from spicy import spatial

vector1 = [1, 2, 3]
vector2 = [3, 2, 1]

cosine_similarity = 1 - spatial.distance.cosine(vector1, vector2)
print(cosine_similarity)



import spacy

nlp = spacy.load("en")


l =["My","name","is","jyothi"]

def permutation(lst):

    # If lst is empty then there are no permutations

    if len(lst) == 0:

        return []

    # If there is only one element in lst then, only

    # one permuatation is possible

    if len(lst) == 1:

        return [lst]

  
    # Find the permutations for lst if there are

    # more than 1 characters

    l = [] # empty list that will store current permutation
    #d=  []
 
    # Iterate the input(lst) and calculate the permutation

    for i in range(len(lst)):

       m = lst[i]

       # Extract lst[i] or m from the list.  remLst is

       # remaining list

       remLst = lst[:i] + lst[i+1:]

       # Generating all permutations where m is first

       # element

       for p in permutation(remLst):
        
           l.append([m] + p)
           l.append(p)

    return(l)

# Driver program to test above function

d =[]
for p in permutation(l):

    d.append(p)

e = ['jyothi', 'is']
e2= ""
for i in e:
    print(i)
    e2 =  e2 +" "+str(i)
    
e2 = nlp(e2)

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
  
# X = input("Enter first string: ").lower() 
# Y = input("Enter second string: ").lower() 
X ="is jyothi..;"
Y ="is jyothi"
  
# tokenization 
X_list = word_tokenize(X)  
Y_list = word_tokenize(Y) 
  
# sw contains the list of stopwords 
sw = stopwords.words('english')  
l1 =[];l2 =[] 
  
# remove stop words from string 
X_set = {w for w in X_list if not w in sw}  
Y_set = {w for w in Y_list if not w in sw} 
  
# form a set containing keywords of both strings  
rvector = X_set.union(Y_set)  
for w in rvector: 
    if w in X_set: l1.append(1) # create a vector 
    else: l1.append(0) 
    if w in Y_set: l2.append(1) 
    else: l2.append(0) 
c = 0
  
# cosine formula  
for i in range(len(rvector)): 
        c+= l1[i]*l2[i] 
cosine = c / float((sum(l1)*sum(l2))**0.5) 
print("similarity: ", cosine) 

if(q==qq):
    qqq=1















