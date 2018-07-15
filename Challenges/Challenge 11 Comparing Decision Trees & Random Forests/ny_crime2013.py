
# coding: utf-8

# ## Build a regression model to predict property crimes, the focus is on creating a model that explains a lot of variance.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("seaborn")


# In[2]:


crime_data = pd.read_excel("table_8_offenses_known_to_law_enforcement_new_york_by_city_2013.xls",skiprows=4)
crime_data.head(5)
#with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    #print(crime_data[["City","Population","Property\ncrime"]])


# In[3]:


# display column headers for possible rename simplification
print(crime_data.columns.values.tolist())


# In[4]:


# rename murder and property crime column
crime_data = crime_data.rename(columns={"Murder and\nnonnegligent\nmanslaughter":"Murder","Property\ncrime":"Property Crime"})


# In[5]:


# create population squared feature for prediction 
crime_data["Population^2"] = crime_data["Population"].apply(lambda x: x**2)


# In[6]:


# found that New York Total was present in data set, need to drop
crime_data = crime_data[crime_data["City"] != "New York"]
crime_data.describe()


# ## Function Specification for Model:
# 
# ### PropertyCrime = x + Population + Population^2 + Murder + Robbery

# In[7]:


crime_model = crime_data[["Population","Murder","Robbery","Property Crime","Population^2"]]
crime_model.head(5)


# In[8]:


# check for NaN values
crime_model.isnull().sum()


# In[9]:


# check which rows contain NaN values (determine scope of missing values)
for col in crime_model.columns:
    print(crime_model[crime_model[col].isnull()][crime_model.columns[crime_model.isnull().any()]])


# In[10]:


# rows appear to be consistent across columns, can remove all rows with NaN
crime_model = crime_model.dropna(how="all")
crime_model.isnull().sum()


# In[11]:


# correlation coefficient between columns
crime_model.corr()


# In[12]:


# visualize data to find possible outliers and distribution of data
sns.pairplot(crime_model)
plt.show()


# In[13]:


# check dtypes of columns to check for compatibility
crime_model.dtypes


# In[14]:


# describe dataset
crime_model.describe()


# In[15]:


# normalize data to adjust visuals
def normalize_data(df):
    x = (df-df.min())/(df.max()-df.min())
    return x

col_list = crime_model.columns.values.tolist()

for col in col_list:
    crime_model[col] = normalize_data(crime_model[col])
    # change 0 values for transformations
    crime_model[col] = crime_model[col].mask(crime_model[col] == 0, 0.00001)
    
crime_model.describe()   


# In[16]:


sns.pairplot(crime_model)
plt.show()


# In[17]:


# appears to be some outliers, use boxplot to get clearer visual
plt.rcParams['figure.figsize'] = (23,13)
fig,ax = plt.subplots(ncols = 5)
y = 0
for col in col_list:
    ax[y].boxplot(crime_model[col],sym="k.")
    ax[y].set_title(col + " Distribution")
    y+=1
plt.show()


# In[18]:


# describe dataset to locate potential outlier
crime_model.describe()


# In[19]:


# appears to be some significant outliers in data, proceed to clean
from scipy import stats
def transform_df(df_col):
    x,lam = stats.boxcox(df_col)
    return x

for col in col_list:
    crime_model[col] = transform_df(crime_model[col])


# In[20]:


sns.pairplot(crime_model)
plt.show()


# In[21]:


# remove significant outliers (more than 3 standard deviations from mean)
crime_model[crime_model.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
crime_model.describe()


# In[22]:


crime_model.isnull().sum()


# In[23]:


sns.pairplot(crime_model)
plt.show()


# In[24]:


fig,ax = plt.subplots(ncols = 5)
y = 0
for col in col_list:
    ax[y].boxplot(crime_model[col],sym="k.")
    ax[y].set_title(col + " Distribution")
    y+=1
plt.show()


# In[25]:


# apply dataframe to linear regression model
crime_regr = linear_model.LinearRegression()
Y = crime_model["Property Crime"]
X = crime_model[["Population","Murder","Robbery"]]
crime_regr.fit(X,Y)

print("Coefficients:\n",crime_regr.coef_)
print("\nIntercept:\n",crime_regr.intercept_)
print("\nR-squared: ")
print(crime_regr.score(X,Y))


# In[26]:


# R-squared value appears extremely high, test for overfitting (holdout groups)
from sklearn.model_selection import cross_val_score
cross_val_score(crime_regr,X,Y,cv=10)

