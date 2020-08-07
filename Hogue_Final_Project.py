#!/usr/bin/env python
# coding: utf-8

# ## Andrew Hogue (amh3ze)
# 
# ## DS5001 Final Project

# ### Data Source
# 
# https://kavita-ganesan.com/entity-ranking-data/#.XxtCdcfQh3g

# ### Import necessary packages

# In[1]:


import os
import pandas as pd
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.stem.porter import PorterStemmer


# ### Create dataframe and pull in reviews

# In[231]:


data = pd.DataFrame(columns = ['Year', 'Make', 'Model', 'Review'])


# In[234]:


for folder in os.listdir(file_dir):
    for fil in os.listdir(file_dir+folder):
        f = open(str(file_dir) + str(folder) + "/" + str(fil), 'r')
        read = f.read()
        if fil == '2007_toyota_camry':
            text=read.split('\n')
            text = list(filter(None, text))
        elif fil == '2008_honda_accord':
            text=read.split('\n')
            text = list(filter(None, text))
            regex = re.compile(r"^\w{1,}")
            filtered = filter(lambda x: regex.search(x), text)
            text = list(filtered)
        else:    
            doc = re.findall(r"<DOCNO>(.*?)</DOCNO>", read)
            text = re.findall(r"<TEXT>(.*?)</TEXT>", read)
        make = str(fil).split('_')[1]
        if len(fil.split('_')) == 3:
            model = str(fil).split('_')[2].strip("']")
        else:
            model = str(fil).split('_')[2:]
            model = ' '.join(model)
        data = data.append({'Year':folder, 'Make':make, 'Model':model,'Review':text}, ignore_index=True)


# In[235]:


data.append({'Year':folder, 'Make':make, 'Model':model,'Review':text}, ignore_index=True)


# In[238]:


for i in range(0, len(data.Review)):
    data.Review[i] = ' '.join(data.Review[i])


# In[240]:


data.to_csv('all_reviews.csv')


# In[ ]:


data = data.explode('Review')


# In[ ]:


#data.to_csv('E:/Creative Cloud Files/DS5001/Project/car_reviews.csv')


# In[38]:


data = pd.read_csv('car_reviews.csv')


# In[42]:


data = data.drop(['Unnamed: 0'], axis=1)


# In[43]:


data


# In[ ]:


bools = data['Review'].isna()
bools.where(bools == True)


# In[13]:


OHCO = ['Year', 'Make', 'Model']


# In[48]:


def tokenize_review(rev):
    rev = rev.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rev)
    filtered_words = [w for w in tokens if not w in set(stopwords.words('english'))]
    #return " ".join(filtered_words)
    return filtered_words


# In[7]:


tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(data['Review'][0].lower())


# In[ ]:


tokens


# In[ ]:


#data['Review'][0].lower()


# In[49]:


revtok = []
for item in data['Review']:
    revtok.append(tokenize_review(item))  


# In[50]:


data['term_str'] = revtok


# In[11]:


data['Make'].value_counts()


# In[12]:


make_year_counts = pd.DataFrame(data.groupby(['Year', 'Make']).size())


# In[13]:


make_year_counts


# In[14]:


data2 = data.explode('term_str')


# In[15]:


data2.set_index(OHCO)


# In[16]:


data.set_index(OHCO)


# ### Sentiment Analysis using VADER at sentence level

# In[117]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')


# In[118]:


sia = SentimentIntensityAnalyzer()


# In[19]:


sample = data['Review'][0]


# In[20]:


sia.polarity_scores(sample)['compound']


# In[119]:


data['sentiment'] = data['Review'].apply(lambda x: sia.polarity_scores(x)['compound'])


# In[22]:


data = data.drop('index', axis=1)


# In[219]:


data.head()


# In[121]:


data3 = pd.DataFrame(data.groupby(['Year', 'Make', 'Model']).sentiment.mean())


# In[122]:


data3.head()


# In[123]:


data3 = data3.reset_index()


# In[124]:


data3.head()


# In[125]:


data4 = pd.DataFrame(data.groupby(['Year', 'Make']).sentiment.mean()).reset_index()


# In[126]:


data4.head()


# In[129]:


plt.figure(figsize=(40,20))
ax1 = sns.barplot(x='Make', y='sentiment', hue='Year', data = data4)
ax1.set_title('Make Sentiments by Year (2007-2009)')
plt.savefig('Make Sentiments by Year (2007-2009).png')


# In[130]:


plt.figure(figsize=(30,10))
ax1 = sns.barplot(x='Make', y='sentiment', data = data4)
plt.savefig('Average Sentiment by Make')


# In[131]:


ax5 = sns.catplot(x='Make', y='sentiment', data=data3, height = 20)
ax5.set_titles('Sentiment Range by Make')
plt.savefig('sentiment_range_by_make.png')


# In[33]:


os.chdir('E:/Creative Cloud Files/DS5001/Project/')
share = pd.read_csv('market_share.csv')


# In[ ]:


def share_sent_plot(make): 
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=data4[data4.Make == str(make)].Year, y=data4[data4.Make == str(make)].sentiment, name="Sentiment"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=share[share.Make == str(make)].Year, y=share[share.Make == str(make)].US_Share, name="US Share"),
        secondary_y=True,
    )

    fig.update_layout(
        title_text="Sentiment vs. Market Share by Year for " + str(make)
    )

    fig.update_xaxes(title_text="Year")

    fig.update_yaxes(title_text="Sentiment", secondary_y=False)
    fig.update_yaxes(title_text="US Market Share", secondary_y=True)
    
    fig.write_image("Share_Sent_{}.png".format(make))


# In[ ]:


#for i in share.Make.unique():
#    share_sent_plot(i)


# In[ ]:


share.Year = share.Year.astype('object')


# In[ ]:





# In[ ]:


data2['pos_tuple'] = nltk.pos_tag(data2.term_str)


# In[6]:


data2 = pd.read_csv('tokens.csv')


# In[7]:


data2.head()


# In[8]:


data2 = data2[~data2.term_str.isin(data2.Model)]


# In[9]:


data2 = data2[~data2.term_str.isin(data2.Make)]


# In[10]:


car_stopwords = ['car', 'truck', 'miles', 'ride', 'cars', 'vehicles']


# In[11]:


data2 = data2[~data2.term_str.isin(car_stopwords)]


# ### Topic Models by Make

# In[14]:


words = data2[data2.pos.str.match(r'^NNS?$')]    .groupby(OHCO[1]).term_str    .apply(lambda x: ' '.join(x))    .to_frame()


# In[15]:


words.head()


# In[34]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA


# In[35]:


n_terms = 50
n_topics = 10
max_iter = 10


# In[36]:


tfv = CountVectorizer(max_features=n_terms, stop_words='english')
tf = tfv.fit_transform(words.term_str)
TERMS = tfv.get_feature_names()


# In[37]:


lda = LDA(n_components=n_topics, max_iter=max_iter, learning_offset=50., random_state=0)


# In[38]:


THETA = pd.DataFrame(lda.fit_transform(tf), index=words.index)
THETA.columns.name = 'topic_id'


# In[39]:


THETA.sample(20).style.background_gradient()


# In[40]:


PHI = pd.DataFrame(lda.components_, columns=TERMS)
PHI.index.name = 'topic_id'
PHI.columns.name  = 'term_str'


# In[41]:


PHI.T.head().style.background_gradient()


# In[42]:


TOPICS = PHI.stack().to_frame().rename(columns={0:'weight'})    .groupby('topic_id')    .apply(lambda x: 
           x.weight.sort_values(ascending=False)\
               .head(10)\
               .reset_index()\
               .drop('topic_id',1)\
               .term_str)


# In[43]:


TOPICS


# In[44]:


TOPICS['label'] = TOPICS.apply(lambda x: str(x.name) + ' ' + ' '.join(x), 1)


# In[45]:


TOPICS['doc_weight_sum'] = THETA.sum()


# In[46]:


TOPICS.sort_values('doc_weight_sum', ascending=True).plot.barh(y='doc_weight_sum', x='label', figsize=(5,10)) 


# In[47]:


topic_cols = [t for t in range(n_topics)]
MAKES = THETA.groupby('Make')[topic_cols].mean().T                                            
MAKES.index.name = 'topic_id'


# In[48]:


MAKES.T


# In[49]:


MAKES['topterms'] = TOPICS[[i for i in range(10)]].apply(lambda x: ' '.join(x), 1)


# In[51]:


MAKES.sort_values('acura', ascending=False).style.background_gradient()


# In[53]:


px.scatter(MAKES.reset_index(), 'bmw', 'toyota', hover_name='topterms', text='topic_id')    .update_traces(mode='text')


# ### TFIDF

# In[195]:


data2 = pd.read_csv('tokens.csv').set_index(OHCO[1:3]).drop(['Unnamed: 0', 'index', 'Review', 'Year'], axis=1)


# In[196]:


data2.head()


# In[241]:


data5 = pd.read_csv('all_reviews.csv')


# In[242]:


data5.head()


# In[259]:


revtok = []
for item in data5['Review']:
    revtok.append(tokenize_review(item))  


# In[260]:


data5['term_str'] = revtok


# In[261]:


data5.head()


# In[264]:


data5 = data5.sort_values('Make')


# In[267]:


data5.to_csv('TERMS.csv')


# In[349]:


data6 = data5.set_index('Make').drop(['Model', 'Year', 'Review', 'Unnamed: 0'], 1).explode('term_str')


# In[350]:


data6 = data6[~data6.term_str.isin(data.Make)]


# In[351]:


data6 = data6[~data6.term_str.isin(data.Model)]


# In[353]:


data6 = data6[~data6.term_str.str.contains('\d', regex=True)]


# In[354]:


data6


# In[355]:


BOW = data6.groupby(OHCO[1:2]+['term_str']).term_str.count()    .to_frame().rename(columns={'term_str':'n'})


# In[356]:


BOW.head()


# In[357]:


DTCM = BOW['n'].unstack().fillna(0).astype('int')
DTCM.head()


# In[358]:


tf_method = 'sum'


# In[359]:


if tf_method == 'sum':
    TF = DTCM.T / DTCM.T.sum()
elif tf_method == 'max':
    TF = DTCM.T / DTCM.T.max()
elif tf_method == 'log':
    TF = np.log10(1 + DTCM.T)
elif tf_method == 'raw':
    TF = DTCM.T
elif tf_method == 'double_norm':
    TF = DTCM.T / DTCM.T.max()
    TF = tf_norm_k + (1 - tf_norm_k) * TF[TF > 0]
elif tf_method == 'binary':
    TF = DTCM.T.astype('bool').astype('int')
TF = TF.T


# In[360]:


TF.head()


# In[361]:


DF = DTCM[DTCM > 0].count()


# In[362]:


DF.head()


# In[363]:


N = DTCM.shape[0]


# In[364]:


idf_method = 'standard'


# In[365]:


if idf_method == 'standard':
    IDF = np.log10(N / DF)
elif idf_method == 'max':
    IDF = np.log10(DF.max() / DF) 
elif idf_method == 'smooth':
    IDF = np.log10((1 + N) / (1 + DF)) + 1


# In[366]:


TFIDF = TF * IDF


# In[ ]:





# In[367]:


TFIDF.head(30)


# In[368]:


TFIDF.to_csv('TFIDF.csv')


# In[369]:


TFIDF.max()


# In[307]:


data6['df'] = DF
data6['idf'] = IDF
data6['tfidf_mean'] = TFIDF[TFIDF > 0].mean().fillna(0)
data6['tfidf_sum'] = TFIDF.sum()
data6['tfidf_median'] = TFIDF[TFIDF > 0].median().fillna(0)
data6['tfidf_max'] = TFIDF.max()


# In[370]:


BOW['tf'] = TF.stack()
BOW['tfidf'] = TFIDF.stack()


# In[388]:


BOW.to_csv('BOW_w_TFIDF.csv')


# In[374]:


def tfidf_make(make):
    
    table = BOW.loc[str(make)].sort_values('tfidf', ascending=False)
    
    tops = table.head(100)

    tops.to_csv('tfidf_{}.csv'.format(make))


# In[375]:


for item in data.Make.unique():
    tfidf_make(item)


# ### Wordclouds by Model

# In[376]:


from wordcloud import WordCloud
data = pd.read_csv('all_reviews.csv')


# In[383]:


text = data.Review[0]


# In[384]:


wordcloud = WordCloud().generate(text)


# In[385]:


plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


# In[386]:


def make_wordcloud(item):
    
    text = data.Review[item]
    
    wordcloud = WordCloud().generate(text)
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('Wordcloud_{}_{}_{}.png'.format(data.Make[item], data.Model[item], data.Year[item]))


# In[387]:


for i in range(0, len(data.Review)):
    make_wordcloud(i)


# In[ ]:




