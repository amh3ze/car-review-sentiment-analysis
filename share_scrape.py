#!/usr/bin/env python
# coding: utf-8

# # Andrew Hogue (amh3ze)
# 
# # Car Market Share Web Scraping

# ### Website
# 
# https://www.goodcarbadcar.net

# In[4]:


from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests


# In[122]:


data = pd.DataFrame(columns = ['Make', 'Year', 'Sales', 'YOY_Change', 'US_Share', 'Share_Change'])


# In[16]:


reviews = pd.read_csv('car_reviews.csv')


# In[84]:


table = soup.find('table', id='table_4')


# In[117]:


makes = set(reviews.Make)
makes.remove('smart')
makes.remove('scion')
makes.remove('suzuki')
makes.remove('pontiac')
makes.remove('saturn')


# In[124]:


for i in makes:
    if i=='mercedes-benz':
        page = requests.get("https://www.goodcarbadcar.net/{}-us-figures/".format(i))
    else:
        page = requests.get("https://www.goodcarbadcar.net/{}-us-sales-figures/".format(i))
    soup = BeautifulSoup(page.text, 'html.parser')
    table = soup.find('table', id='table_4')
    rows = table.find_all('tr')
    for j in range(3, 7):
        lines = list(rows[j])
        year = int(lines[1].get_text())
        sales = lines[3].get_text()
        ychange = lines[5].get_text()
        share = lines[7].get_text()
        schange = lines[9].get_text()
        data = data.append({'Make': i, 'Year':year, 'Sales':sales, 'YOY_Change':ychange, 'US_Share':share, 'Share_Change':schange}, ignore_index=True)


# In[126]:


#data.to_csv('market_share.csv')


# In[ ]:




