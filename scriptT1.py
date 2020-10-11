#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import nltk
import string
import matplotlib.pyplot as plt
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('wordnet')

from collections import Counter
from wordcloud import WordCloud
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text

t1 = open('T1.txt', encoding='utf8')
rawt1 = t1.read()
#pre-processing
# lowering of case
rawt1 = rawt1.lower()
# removing of urls
rawt1 = re.sub(r'(?:(https|http)?:\/\/)*','', rawt1, flags=re.MULTILINE)
# removing of websites
rawt1 = re.sub(r'(www.[a-z]*.[a-z]*)','', rawt1)
# removing digits
rawt1 = re.sub(r'[\d]*','', rawt1)
# removing chapter names
rawt1 = re.sub(r'(i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)\.[ _a-z:]*','', rawt1)
# removing punctuations
rawt1 = remove_punctuation(rawt1)
t1_tokenized = word_tokenize(rawt1)
counts = Counter(t1_tokenized)
print("Number of distinct words "+str(len(counts)))
print("Number of tokens "+str(len(t1_tokenized)))
print("Number of characters "+str(len(rawt1)))
print(t1_tokenized)
fdist = FreqDist(t1_tokenized)
fdist.plot(30, cumulative=False)
plt.show()


# In[2]:


word_cloud_dict=Counter(t1_tokenized)
wordcloud = WordCloud(width = 1000, height = 1000,
                      background_color = 'white',
                      stopwords = None).generate_from_frequencies(word_cloud_dict)
plt.figure(figsize = (8,8) , facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()


# In[3]:


stop_words = set(stopwords.words("english"))
filtered_text = []
for w in t1_tokenized:
    if w not in stop_words:
        filtered_text.append(w)
print("Filtered text tokens")
print(len(filtered_text))
print("\nOriginal Tokens")
print(len(t1_tokenized))


# In[4]:


word_cloud_dict=Counter(filtered_text)
wordcloud = WordCloud(width = 1000, height = 1000, background_color = 'white').generate_from_frequencies(word_cloud_dict)
plt.figure(figsize = (8,8) , facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()


# In[5]:


wlen_dist = FreqDist()
for w in filtered_text:
    wlen_dist[len(w)] += 1
wlen_dist.plot(100, cumulative=False)
plt.show()


# In[6]:


# Stemming
ps = PorterStemmer()
stemmed = []
for w in filtered_text:
    stemmed.append(ps.stem(w))
print("No of tokens after stemming")
print(stemmed)


# In[7]:


# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = []
for w in filtered_text:
    lemmatized.append(lemmatizer.lemmatize(w, pos='v'))
print("No of tokens after lemmatizing")
print(lemmatized)


# In[8]:


tagged = nltk.pos_tag(filtered_text)
print(tagged)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




