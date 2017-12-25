

```python
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
```




<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>



# <center>Karthick Venkatesan</center>
### <center>karvenka@umail.iu.edu</center>

## 1. Introduction

Airbnb is an online marketplace and hospitality service, enabling people to lease or rent short-term lodging including vacation rentals, apartment rentals, homestays, hostel beds, or hotel rooms. In order for Airbnb to provide a personalized experience for its customers, it
has explored the possibility of predicting the country destination in which a user will make a booking.
With this information, Airbnb can create more personalized content with its member community, decrease
the average time to first booking, and better forecast demand. These goals provide mutual benefit to
Airbnb and its customers: personal recommendations can improve customer engagement with the
platform, thus encouraging both repeat bookings and referrals to Airbnb for those in a customer’s network
of close friends and family.

This report details the analysis and results  for [Airbnb New User Bookings challenge][1] which was run in Kaggle. In the competition the participants were provided with data from Airbnb customers  and the task is to predict the first
booking destination for new Airbnb customers travelling from the United States. The response variable is
the destination where the booking is made. This can be one of 12 possible values: ' US', 'FR', 'CA', 'GB',
'ES', 'IT', 'PT', 'NL',' DE', 'AU', 'NDF', and 'other'. All of these correspond to two letter country
abbreviations with the exception of ‘NDF’, which corresponds to ‘No destination found’ and implies that
the user has not made a booking.

[1]: https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings

The evaluation metric for this competition is NDCG (Normalized discounted cumulative gain) @k where k=5. The details of the NDCG calculation are available [here][1].

[1]: https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings#evaluation

## 2. Dataset

### Description of the Dataset
The Airbnb challenege has the below datasets - a list of users along with their demographics, web session records, and some summary statistics. We need to predict which country a new user's first booking destination will be. All the users in the dataset are from the USA.

There are 12 possible outcomes of the destination country: 'US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL','DE', 'AU', 'NDF' (no destination found), and 'other'. 'other' means there was a booking, but is to a country not included in the list, while 'NDF' means there wasn't a booking.

The training and test sets are split by dates. The test set contains new users with first activities after 7/1/2014. In the sessions dataset, the data only dates back to 1/1/2014, while the users dataset dates back to 2010. 

### User Dataset

1. __train_users.csv__ - The training set of users
2. __test_users.csv__ - the test set of users

   - id: user id
   - date_account_created: the date of account creation
   - timestamp_first_active: timestamp of the first activity, note that it can be earlier than date_account_created or date_first_booking because a user can search before signing up
   - date_first_booking: date of first booking
   - gender
   - age
   - signup_method
   - signup_flow: the page a user came to signup up from
   - language: international language preference
   - affiliate_channel: what kind of paid marketing
   - affiliate_provider: where the marketing is e.g. google, craigslist, other
   - first_affiliate_tracked: whats the first marketing the user interacted with before the signing up
   - signup_app
   - first_device_type
   - first_browser
   - country_destination: this is the target variable you are to predict

### Session Dataset
1. __sessions.csv__ - web sessions log for users
  - user_id: to be joined with the column 'id' in users table
  - action
  - action_type
  - action_detail
  - device_type
  - secs_elapsed

### Countries Dataset
1. __countries.csv__ - Summary statistics of destination countries in this dataset and their locations
2. __age_gender_bkts.csv__ - Summary statistics of users' age group, gender, country of destination


```python
# Draw inline
%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Set figure aesthetics
sns.set_style("white", {'ytick.major.size': 10.0})
sns.set_context("poster", font_scale=1.1)
```

### Data Exploration

As part Data Exploration step we are looking for answers to the below questions
 - Is there any mistakes in the data?
 - Does the data have peculiar behavior?
 - Do we need to fix or remove any of the data to be more realistic?
 - Do the features in the data capture the variations in our target variable?


```python
# Load the data into DataFrames
path = '../data/'
train_users = pd.read_csv(path + 'train_users_2.csv')
test_users = pd.read_csv(path + 'test_users.csv')
sessions = pd.read_csv(path + 'sessions.csv')
countries = pd.read_csv(path + 'countries.csv')
age_gender = pd.read_csv(path + 'age_gender_bkts.csv')
```

#### Counts


```python
print("We have", train_users.shape[0], "users in the training set and", 
      test_users.shape[0], "in the test set.")
print("In total we have", train_users.shape[0] + test_users.shape[0], "users.")
print("We have", sessions.shape[0], "Session Records for" , sessions.user_id.nunique() , "users." )
print("We have", (train_users.shape[0] + test_users.shape[0] -sessions.user_id.nunique()) , "users with no session records." )
print("We have", (countries.shape[0]) , "records in the countries dataset." )
print("We have", (age_gender.shape[0]) , "records in the age/gender dataset." )
```

    We have 213451 users in the training set and 62096 in the test set.
    In total we have 275547 users.
    We have 10567737 Session Records for 135483 users.
    We have 140064 users with no session records.
    We have 10 records in the countries dataset.
    We have 420 records in the age/gender dataset.


#### Users - Preview


```python
# Merge train and test users
users = pd.concat((train_users, test_users), axis=0, ignore_index=True)

# Remove ID's since now we are not interested in making predictions
users.set_index('id',inplace=True)

users.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>affiliate_channel</th>
      <th>affiliate_provider</th>
      <th>age</th>
      <th>country_destination</th>
      <th>date_account_created</th>
      <th>date_first_booking</th>
      <th>first_affiliate_tracked</th>
      <th>first_browser</th>
      <th>first_device_type</th>
      <th>gender</th>
      <th>language</th>
      <th>signup_app</th>
      <th>signup_flow</th>
      <th>signup_method</th>
      <th>timestamp_first_active</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gxn3p5htnn</th>
      <td>direct</td>
      <td>direct</td>
      <td>NaN</td>
      <td>NDF</td>
      <td>2010-06-28</td>
      <td>NaN</td>
      <td>untracked</td>
      <td>Chrome</td>
      <td>Mac Desktop</td>
      <td>-unknown-</td>
      <td>en</td>
      <td>Web</td>
      <td>0</td>
      <td>facebook</td>
      <td>20090319043255</td>
    </tr>
    <tr>
      <th>820tgsjxq7</th>
      <td>seo</td>
      <td>google</td>
      <td>38.0</td>
      <td>NDF</td>
      <td>2011-05-25</td>
      <td>NaN</td>
      <td>untracked</td>
      <td>Chrome</td>
      <td>Mac Desktop</td>
      <td>MALE</td>
      <td>en</td>
      <td>Web</td>
      <td>0</td>
      <td>facebook</td>
      <td>20090523174809</td>
    </tr>
    <tr>
      <th>4ft3gnwmtx</th>
      <td>direct</td>
      <td>direct</td>
      <td>56.0</td>
      <td>US</td>
      <td>2010-09-28</td>
      <td>2010-08-02</td>
      <td>untracked</td>
      <td>IE</td>
      <td>Windows Desktop</td>
      <td>FEMALE</td>
      <td>en</td>
      <td>Web</td>
      <td>3</td>
      <td>basic</td>
      <td>20090609231247</td>
    </tr>
    <tr>
      <th>bjjt8pjhuk</th>
      <td>direct</td>
      <td>direct</td>
      <td>42.0</td>
      <td>other</td>
      <td>2011-12-05</td>
      <td>2012-09-08</td>
      <td>untracked</td>
      <td>Firefox</td>
      <td>Mac Desktop</td>
      <td>FEMALE</td>
      <td>en</td>
      <td>Web</td>
      <td>0</td>
      <td>facebook</td>
      <td>20091031060129</td>
    </tr>
    <tr>
      <th>87mebub9p4</th>
      <td>direct</td>
      <td>direct</td>
      <td>41.0</td>
      <td>US</td>
      <td>2010-09-14</td>
      <td>2010-02-18</td>
      <td>untracked</td>
      <td>Chrome</td>
      <td>Mac Desktop</td>
      <td>-unknown-</td>
      <td>en</td>
      <td>Web</td>
      <td>0</td>
      <td>basic</td>
      <td>20091208061105</td>
    </tr>
  </tbody>
</table>
</div>



#### Sessions - Preview


```python
sessions.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>action</th>
      <th>action_type</th>
      <th>action_detail</th>
      <th>device_type</th>
      <th>secs_elapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>d1mm9tcy42</td>
      <td>lookup</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Windows Desktop</td>
      <td>319.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d1mm9tcy42</td>
      <td>search_results</td>
      <td>click</td>
      <td>view_search_results</td>
      <td>Windows Desktop</td>
      <td>67753.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d1mm9tcy42</td>
      <td>lookup</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Windows Desktop</td>
      <td>301.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d1mm9tcy42</td>
      <td>search_results</td>
      <td>click</td>
      <td>view_search_results</td>
      <td>Windows Desktop</td>
      <td>22141.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>d1mm9tcy42</td>
      <td>lookup</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Windows Desktop</td>
      <td>435.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Countries - Preview


```python
countries
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_destination</th>
      <th>lat_destination</th>
      <th>lng_destination</th>
      <th>distance_km</th>
      <th>destination_km2</th>
      <th>destination_language</th>
      <th>language_levenshtein_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AU</td>
      <td>-26.853388</td>
      <td>133.275160</td>
      <td>15297.7440</td>
      <td>7741220.0</td>
      <td>eng</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CA</td>
      <td>62.393303</td>
      <td>-96.818146</td>
      <td>2828.1333</td>
      <td>9984670.0</td>
      <td>eng</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DE</td>
      <td>51.165707</td>
      <td>10.452764</td>
      <td>7879.5680</td>
      <td>357022.0</td>
      <td>deu</td>
      <td>72.61</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ES</td>
      <td>39.896027</td>
      <td>-2.487694</td>
      <td>7730.7240</td>
      <td>505370.0</td>
      <td>spa</td>
      <td>92.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FR</td>
      <td>46.232193</td>
      <td>2.209667</td>
      <td>7682.9450</td>
      <td>643801.0</td>
      <td>fra</td>
      <td>92.06</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GB</td>
      <td>54.633220</td>
      <td>-3.432277</td>
      <td>6883.6590</td>
      <td>243610.0</td>
      <td>eng</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>IT</td>
      <td>41.873990</td>
      <td>12.564167</td>
      <td>8636.6310</td>
      <td>301340.0</td>
      <td>ita</td>
      <td>89.40</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NL</td>
      <td>52.133057</td>
      <td>5.295250</td>
      <td>7524.3203</td>
      <td>41543.0</td>
      <td>nld</td>
      <td>63.22</td>
    </tr>
    <tr>
      <th>8</th>
      <td>PT</td>
      <td>39.553444</td>
      <td>-7.839319</td>
      <td>7355.2534</td>
      <td>92090.0</td>
      <td>por</td>
      <td>95.45</td>
    </tr>
    <tr>
      <th>9</th>
      <td>US</td>
      <td>36.966427</td>
      <td>-95.844030</td>
      <td>0.0000</td>
      <td>9826675.0</td>
      <td>eng</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



#### Age/Gender - Preview


```python
age_gender.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age_bucket</th>
      <th>country_destination</th>
      <th>gender</th>
      <th>population_in_thousands</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100+</td>
      <td>AU</td>
      <td>male</td>
      <td>1.0</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>95-99</td>
      <td>AU</td>
      <td>male</td>
      <td>9.0</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>90-94</td>
      <td>AU</td>
      <td>male</td>
      <td>47.0</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>85-89</td>
      <td>AU</td>
      <td>male</td>
      <td>118.0</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>80-84</td>
      <td>AU</td>
      <td>male</td>
      <td>199.0</td>
      <td>2015.0</td>
    </tr>
  </tbody>
</table>
</div>



### Missing Data

Usually the missing data comes in the way of *NaN*, but in the data above above we can see that the `gender` column some values being `-unknown-`. We transformed those values into *NaN* first and then summarised the percentage of unknowns in each field. 


```python
users.gender.replace('-unknown-', np.nan, inplace=True)
users.first_browser.replace('-unknown-', np.nan, inplace=True)
```


```python
users_nan = (users.isnull().sum() / users.shape[0]) * 100
users_nan[users_nan > 0].drop('country_destination')
```




    age                        42.412365
    date_first_booking         67.733998
    first_affiliate_tracked     2.208335
    first_browser              16.111226
    gender                     46.990169
    dtype: float64



We have quite a lot of *NaN* in the `age` and `gender` wich will yield in lesser performance of the classifiers we will build. The feature `date_first_booking` has a 67% of NaN values and this feature is not present for the tests users.

The other feature with a high rate of *NaN* was `age`. 


```python
users.age.describe()
```




    count    158681.000000
    mean         47.145310
    std         142.629468
    min           1.000000
    25%          28.000000
    50%          33.000000
    75%          42.000000
    max        2014.000000
    Name: age, dtype: float64



There is some inconsistency in the age of some users as we can see above. It could be because the `age` inpout field was not sanitized or there was some mistakes handlig the data.


```python
print('Users with age greater than 85 : ' + str(sum(users.age > 85)))
print('Users with age less than 18 : ' + str(sum(users.age < 18)))
```

    Users with age greater than 85 : 3041
    Users with age less than 18 : 188


Looking at the data with age more than 85 we can see that these seem to be year values . So these are possibly user entry errors which the system didnt validate.


```python
users[users.age > 85]['age'].describe()
```




    count    3041.000000
    mean      621.953963
    std       847.508105
    min        86.000000
    25%       105.000000
    50%       105.000000
    75%      1953.000000
    max      2014.000000
    Name: age, dtype: float64



The values below 18 summarized below also seem to be user entry errors which needs to be corrected . We have set these values to Nan so that they dont adversely affect our models.


```python
users[users.age < 18]['age'].describe()
```




    count    188.000000
    mean      12.718085
    std        5.764569
    min        1.000000
    25%        5.000000
    50%       16.000000
    75%       17.000000
    max       17.000000
    Name: age, dtype: float64




```python
users.loc[users.age > 85, 'age'] = np.nan
users.loc[users.age < 18, 'age'] = np.nan
```

### Data Types

In the next step we converted each feature as what they are. We transformed the date and categorical variables into the corresponding datatypes.

__Categorical Data:__
   - affiliate_channel
   - affiliate_provider
   - country_destination
   - first_affiliate_tracked
   - first_browser
   - first_device_type
   - gender
   - language
   - signup_app
   - signup_method

__Date Date:__
   - date_account_created
   - date_first_booking
   - date_first_active


```python
categorical_features = [
    'affiliate_channel',
    'affiliate_provider',
    'country_destination',
    'first_affiliate_tracked',
    'first_browser',
    'first_device_type',
    'gender',
    'language',
    'signup_app',
    'signup_method'
]

for categorical_feature in categorical_features:
    users[categorical_feature] = users[categorical_feature].astype('category')
```


```python
users['date_account_created'] = pd.to_datetime(users['date_account_created'])
users['date_first_booking'] = pd.to_datetime(users['date_first_booking'])
users['date_first_active'] = pd.to_datetime(users['timestamp_first_active'], format='%Y%m%d%H%M%S')
```

### Visualizing the Data

#### Gender


```python
users.gender.value_counts(dropna=False).plot(kind='bar', color='#FD5C64', rot=0)
plt.xlabel('Gender')
sns.despine()
```


![png](Venkatesan_Karthick_Final_Project_Report_files/Venkatesan_Karthick_Final_Project_Report_45_0.png)


The above plot helps us to visuvalize the amount of missing data for this feature. We can also notice that there is a slight difference in the counts between the user gender.

Next thing we looked at is to see if there is any gender preferences when travelling.As we can see in the plot there are no big differences between the 2 main genders.


```python
women = sum(users['gender'] == 'FEMALE')
men = sum(users['gender'] == 'MALE')

female_destinations = users.loc[users['gender'] == 'FEMALE', 'country_destination'].value_counts() / women * 100
male_destinations = users.loc[users['gender'] == 'MALE', 'country_destination'].value_counts() / men * 100

# Bar width
width = 0.4

male_destinations.plot(kind='bar', width=width, color='#4DD3C9', position=0, label='Male', rot=0)
female_destinations.plot(kind='bar', width=width, color='#FFA35D', position=1, label='Female', rot=0)

plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage')

sns.despine()
plt.show()
```


![png](Venkatesan_Karthick_Final_Project_Report_files/Venkatesan_Karthick_Final_Project_Report_47_0.png)


#### Country Destination

We plotted the counts of the country destination . As seen in the figure below nearly 60% of the customers end up not making a booking. Among the customers who do end up making a booking US is the preffered destination for more than 2/3 rd's.


```python
counts =  users.country_destination.value_counts(normalize=True).plot(kind='bar')
plt.xlabel('Destination Country')
plt.ylabel('Percentage')
```




    <matplotlib.text.Text at 0x7fc6d465c4e0>




![png](Venkatesan_Karthick_Final_Project_Report_files/Venkatesan_Karthick_Final_Project_Report_50_1.png)


#### Age

The plot of the age data of the users in the traning and test data is as below.


```python
sns.distplot(users.age.dropna(), color='#FD5C64')
plt.xlabel('Age')
sns.despine()
```


![png](Venkatesan_Karthick_Final_Project_Report_files/Venkatesan_Karthick_Final_Project_Report_53_0.png)


As we would expect, the common age to travel is between 25 and 40. We wanted to further explore if there is difference in the booking patterns based on the age of the users. We took a arbitratry split range of 50 and plotted the below graph .


```python
age = 50

younger = sum(users.loc[users['age'] < age, 'country_destination'].value_counts())
older = sum(users.loc[users['age'] > age, 'country_destination'].value_counts())

younger_destinations = users.loc[users['age'] < age, 'country_destination'].value_counts() / younger * 100
older_destinations = users.loc[users['age'] > age, 'country_destination'].value_counts() / older * 100

younger_destinations.plot(kind='bar', width=width, color='#63EA55', position=0, label='Youngers', rot=0)
older_destinations.plot(kind='bar', width=width, color='#4DD3C9', position=1, label='Olders', rot=0)

plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage')

sns.despine()
plt.show()
```


![png](Venkatesan_Karthick_Final_Project_Report_files/Venkatesan_Karthick_Final_Project_Report_55_0.png)


We can see that the young people tend to stay in the US, and the older people choose to travel outside the country. 

### Language
We explore the language feature to understand the distribution and see it would make a good predictor for destination country.We can visuvalize below that language does capture variations in booking destination of the users. For example in the plot below for people with 'fr' language country destnation 'fr' is the second most preffered destination for first booking after 'US'.


```python
import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0,1,22))
users[~(users['country_destination'].isin(['NDF']))].groupby(['country_destination' , 'language']).size().unstack().plot(kind='bar', figsize=(20,10),stacked=False,color=colors)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          ncol=1, fancybox=True, shadow=True)
plt.yscale('log')
plt.xlabel('Destination Country')
plt.ylabel('Log(Count)')
```




    <matplotlib.text.Text at 0x7fc6d03d2198>




![png](Venkatesan_Karthick_Final_Project_Report_files/Venkatesan_Karthick_Final_Project_Report_58_1.png)


#### Dates

We next explore the date account created feature and plot the count of users created by date.


```python
sns.set_style("whitegrid", {'axes.edgecolor': '0'})
sns.set_context("poster", font_scale=1.1)
users.date_account_created.value_counts().plot(kind='line', linewidth=1.2, color='#FD5C64')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc6cf257ba8>




![png](Venkatesan_Karthick_Final_Project_Report_files/Venkatesan_Karthick_Final_Project_Report_61_1.png)


We can visuvalize based on the graph how fast Airbnb has grown from 2012. We next plot the first time active field to see if the feature correlates with the date of user creation . We can see from the below graph there is a very close correlation between the two date fields.


```python
date_first_active = users.date_first_active.apply(lambda x: datetime.datetime(x.year, x.month, x.day))
date_first_active.value_counts().plot(kind='line', linewidth=1.2, color='#FD5C64')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc6cf234128>




![png](Venkatesan_Karthick_Final_Project_Report_files/Venkatesan_Karthick_Final_Project_Report_63_1.png)



```python
users['date_account_created'] = pd.to_datetime(users['date_account_created'], errors='ignore')
users['date_first_active'] = pd.to_datetime(users['timestamp_first_active'], format='%Y%m%d%H%M%S')
users['date_first_booking'] = pd.to_datetime(users['date_first_booking'], errors='ignore')
```

We see minor ups and down in the date plot to understand this variation better we plotted the count of number of users who signed up in Airbnb for each month. We can see that  is a general upward trend in the number of users that are created however we can aslo see there is a pattern in the number of users that sign up for each month possibly a indication of an underlying trend. This is something this we would anticipate for Airbnb as people would tend to travel more during summer and holidays and travel less in other months.


```python
df = users[~users['country_destination'].isnull()]
df.groupby([df["date_account_created"].dt.year, df["date_account_created"].dt.month])['country_destination'].count().plot(kind="bar",figsize=(20,10))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc6cf276ef0>




![png](Venkatesan_Karthick_Final_Project_Report_files/Venkatesan_Karthick_Final_Project_Report_66_1.png)


We further explore the month dependency by plotting the month of booking vs country destinations. We can see from the plot below that inaddition to the variations in the total number of bookings by month we also see a variations in the destination that was booked.
For example from the plot below Australia has significantly higher bookings in months 11 and 12 (November and December) compared to other months.


```python
import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0,1,12))
df[df["date_first_booking"].dt.year == 2013].groupby(['country_destination' , df["date_first_booking"].dt.month]).size().unstack().plot(kind='bar', stacked=False,color=colors)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          ncol=1, fancybox=True, shadow=True)
plt.yscale('log')
plt.xlabel('Destination Country by Month 2013')
plt.ylabel('Log(Count)')
```




    <matplotlib.text.Text at 0x7fc6ceccd7f0>




![png](Venkatesan_Karthick_Final_Project_Report_files/Venkatesan_Karthick_Final_Project_Report_68_1.png)


### Affiliate Information

Below is a plot of the number of bookings per destination by affiliate channel/affiliate provider/first affiliate tracked. We can see a disernable pattern . So all these three features are good predictors of the destination country.


```python
colors = cm.rainbow(np.linspace(0,1,users['affiliate_channel'].nunique()))
users.groupby(['country_destination','affiliate_channel']).size().unstack().plot(kind='bar', stacked=False,color=colors)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          ncol=1, fancybox=True, shadow=True)
plt.yscale('log')
plt.xlabel('Destination Country by affiliate channel')
plt.ylabel('Log(Count)')
```




    <matplotlib.text.Text at 0x7fc6ce6c80b8>




![png](Venkatesan_Karthick_Final_Project_Report_files/Venkatesan_Karthick_Final_Project_Report_71_1.png)



```python
colors = cm.rainbow(np.linspace(0,1,users['affiliate_provider'].nunique()))
users.groupby(['country_destination','affiliate_provider']).size().unstack().plot(kind='bar', stacked=False,color=colors)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          ncol=1, fancybox=True, shadow=True)
plt.yscale('log')
plt.xlabel('Destination Country by affiliate provider')
plt.ylabel('Log(Count)')
```




    <matplotlib.text.Text at 0x7fc6ccabe2b0>




![png](Venkatesan_Karthick_Final_Project_Report_files/Venkatesan_Karthick_Final_Project_Report_72_1.png)



```python
colors = cm.rainbow(np.linspace(0,1,users['first_affiliate_tracked'].nunique()))
users.groupby(['country_destination','first_affiliate_tracked']).size().unstack().plot(kind='bar', stacked=False,color=colors)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          ncol=1, fancybox=True, shadow=True)
plt.yscale('log')
plt.xlabel('Destination Country by first affiliate tracked')
plt.ylabel('Log(Count)')
```




    <matplotlib.text.Text at 0x7fc6cc4c7748>




![png](Venkatesan_Karthick_Final_Project_Report_files/Venkatesan_Karthick_Final_Project_Report_73_1.png)


## 3. Preprocessing

## Age
In the data exploration we already noted that the Age was key feature for predicting destinations. We are also aware of the outiers ie values greater than 85 and less than 18 . In this step we clean the outliers so that they dont adversely affect our model. 

Also , Age is really fine grained. We are going to make bins and fit each user in the proper age group so that we can predict it in categories. Below is a plot of the numbers of users in each Age group . Group 0 is largest which represents the users with no age data.


```python
import numpy as np
import pandas as pd
users.loc[users.age > 85, 'age'] = np.nan
users.loc[users.age < 18, 'age'] = np.nan
users['age'].fillna(-1,inplace=True)
bins = [-1, 0, 4, 9, 14, 19, 24, 29, 34,39,44,49,54,59,64,69,74,79,84,89]
users['age_group'] = np.digitize(users['age'], bins, right=True)
```


```python
%matplotlib inline
users.age_group.value_counts().plot(kind='bar')
plt.yscale('log')
plt.xlabel('Age Group')
plt.ylabel('Log(Count)')
```




    <matplotlib.text.Text at 0x7fc6cb9d66a0>




![png](Venkatesan_Karthick_Final_Project_Report_files/Venkatesan_Karthick_Final_Project_Report_77_1.png)


## Date
We first cast the date records to proper date format. We had noted in the exploration that the month in which either the user signed up in Airbnb or booked in Airbnb is good predictor for the destination the user will book . So from the create date we parse out the month , week day as seperate features. We also create new features based on the date first active field which includes a time stamp. We create month first active , week day first active variable. In addition to this we also create a hour first active as we hypotesize that someone creating a account for the first time at odd hours are more likely to make a booking . 


```python
df = users[users['country_destination'].isnull()]
```


```python
date_account_created = pd.DatetimeIndex(users['date_account_created'])
date_first_active = pd.DatetimeIndex(users['date_first_active'])
date_first_booking = pd.DatetimeIndex(users['date_first_booking'])
```


```python
#users['day_account_created'] = date_account_created.day
users['weekday_account_created'] = date_account_created.weekday
#users['week_account_created'] = date_account_created.week
users['month_account_created'] = date_account_created.month
#users['year_account_created'] = date_account_created.year
#users['day_first_active'] = date_first_active.day
users['weekday_first_active'] = date_first_active.weekday
#users['week_first_active'] = date_first_active.week
users['month_first_active'] = date_first_active.month
users['month_first_book'] = date_first_booking.month
users['hour_first_active'] = date_first_active.hour
#users['year_first_active'] = date_first_active.year
```


```python
users['time_lag_create'] = (date_first_booking - date_account_created).days
users['time_lag_active'] = (date_first_booking - date_first_active).days
users['time_lag_create'].fillna(365,inplace=True)
users['time_lag_active'].fillna(365,inplace=True)
```

Next we create two new features from the dates one is the __time lag create__ which is the time in days between the date of creation and date the customer made the first booking in Airbnb and the second one is __time lag active__ which is the time in days between the date the customer was first active to the date the customer made the first booking . The below plot's show that the time lag features will be able to effectively predict both whether a customer will make a booking or not and also the ultimate destination the customer will make the booking.


```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import importlib
importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)
ax = sns.boxplot(x="country_destination", y="time_lag_create", showfliers=False,data=users[~(users['country_destination'].isnull())])
#users[~(users['country_destination'].isnull())][['time_lag_create','country_destination']].boxplot(by='country_destination')
```


![png](Venkatesan_Karthick_Final_Project_Report_files/Venkatesan_Karthick_Final_Project_Report_84_0.png)



```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import importlib
importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)
ax = sns.boxplot(x="country_destination", y="time_lag_active", showfliers=False,data=users[~(users['country_destination'].isnull())])
#users[~(users['country_destination'].isnull())][['time_lag_create','country_destination']].boxplot(by='country_destination')
```


![png](Venkatesan_Karthick_Final_Project_Report_files/Venkatesan_Karthick_Final_Project_Report_85_0.png)



```python
users[['time_lag_create','time_lag_active']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time_lag_create</th>
      <th>time_lag_active</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>275547.000000</td>
      <td>275547.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>261.543098</td>
      <td>261.326812</td>
    </tr>
    <tr>
      <th>std</th>
      <td>157.921613</td>
      <td>158.395423</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-349.000000</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>41.000000</td>
      <td>41.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>365.000000</td>
      <td>365.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>365.000000</td>
      <td>365.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>365.000000</td>
      <td>1368.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
users.loc[users.time_lag_create > 365, 'time_lag_create'] = 365
users.loc[users.time_lag_active > 365, 'time_lag_create'] = 365
```


```python
drop_list = [
    'date_account_created',
    'date_first_active',
    'date_first_booking',
    'timestamp_first_active',
    'age'
]

users.drop(drop_list, axis=1, inplace=True)
```

## Session Information

There is a lot of information in the `sessions.csv` file. We extracted the below features from the sessions.csv file.


   - Count of each action types
   - Sum of time elapse for each action
   - Pertange of the time elapsed for each action type aginst the total elapsed time for a user
   - Count of unique number action detail
   - Count of number uniqe devices the user used

We merged all this data back with the users data. We had to note that the sessions file included data only from 1/1/2014 however the training data included users from 2010 so there were a large number of user records who dont have any session data . However for all test users we have the session data so we would expect the model do probably less efectively on the training set based on how we choose data to train and test but the model will probably do better on the the testing data.


```python
sessions.rename(columns = {'user_id': 'id'}, inplace=True)
```


```python
from sklearn import preprocessing
# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

action_count = sessions.groupby(['id'])['action'].nunique()

#action_count = pd.DataFrame(min_max_scaler.fit_transform(action_count.fillna(0)),columns=action_count.columns)
action_type_count = sessions.groupby(['id', 'action_type'])['secs_elapsed'].agg(len).unstack()
action_type_count.columns = action_type_count.columns.map(lambda x: str(x) + '_count')
#action_type_count = pd.DataFrame(min_max_scaler.fit_transform(action_type_count.fillna(0)),columns=action_type_count.columns)
action_type_sum = sessions.groupby(['id', 'action_type'])['secs_elapsed'].agg(sum)

action_type_pcts = action_type_sum.groupby(level=0).apply(lambda x:
                                                 100 * x / float(x.sum())).unstack()
action_type_pcts.columns = action_type_pcts.columns.map(lambda x: str(x) + '_pct')
action_type_sum = action_type_sum.unstack()
action_type_sum.columns = action_type_sum.columns.map(lambda x: str(x) + '_sum')
action_detail_count = sessions.groupby(['id'])['action_detail'].nunique()

#action_detail_count = pd.DataFrame(min_max_scaler.fit_transform(action_detail_count.fillna(0)),columns=action_detail_count.columns)

device_type_sum = sessions.groupby(['id'])['device_type'].nunique()

#device_type_sum = pd.DataFrame(min_max_scaler.fit_transform(device_type_sum.fillna(0)),columns=device_type_sum.columns)

sessions_data = pd.concat([action_count, action_type_count, action_type_sum,action_type_pcts,action_detail_count, device_type_sum],axis=1)
action_count = None
action_type_count = None
action_detail_count = None
device_type_sum = None


#users = users.join(sessions_data, on='id')
```


```python
users= users.reset_index().join(sessions_data, on='id')
```

### Encode categorical features

The next step is to encode categorical features. The categorical variables in the data cannot be used as is in the machine learning models. We need to encode them into numeric values. We encoded the categorical variables using the __one hot encoding__ method. 


```python
from sklearn.preprocessing import LabelEncoder
categorical_features = [
    'gender', 'signup_method', 'signup_flow', 'language',
    'affiliate_channel', 'age_group','weekday_account_created','month_account_created','weekday_first_active','month_first_active','hour_first_active',
    'signup_app','affiliate_provider', 'first_affiliate_tracked','first_device_type', 'first_browser'
]
users_sc = users.copy(deep=True)
encode = LabelEncoder()
for j in categorical_features:
    users_sc[j] = encode.fit_transform(users[j].astype('str'))
```

### Feature Selection

At the end of all preprocessing steps we had a total of 54 features that we have extracted from the users and sessions data set. Using all these features may lead to overfitting and could also be time consuming processing on many Machine learning algorithms. So in the next step we performed feature selection were we reduced the number of features using a standard sklearn  feature extraction library __VarianceThreshold__.

__VarianceThreshold__ is a simple baseline approach to feature selection. It removes all features whose variance doesn’t meet some threshold. By default, it removes all zero-variance features, i.e. features that have the same value in all samples.
As an example, suppose that we have a dataset with boolean features, and we want to remove all features that are either one or zero (on or off) in more than 80% of the samples. 

These features which are common for more than 80 % of the data dont necessarily capture all the avriations in the data and will not be good predictors . So we eliminate the variables with low threshold and arrived at a final list of features which are listed below.

Below is the final list of 43 features which we used to build our models.


```python
colx = users_sc.columns.tolist()
rm_list = ['id','country_destination']
for x in rm_list:
    colx.remove(x)
X = users_sc[~(users_sc['country_destination'].isnull())][colx]
X.fillna(0,inplace=True)
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(0.8))
sel.fit_transform(X)
idxs = sel.get_support(indices=True)
colo = [X.columns.tolist()[i] for i in idxs]
print ('\n'.join(colo))
for y in rm_list:
    colo.append(y)
```

    affiliate_channel
    affiliate_provider
    first_affiliate_tracked
    first_browser
    first_device_type
    gender
    language
    signup_flow
    age_group
    weekday_account_created
    month_account_created
    weekday_first_active
    month_first_active
    month_first_book
    hour_first_active
    time_lag_create
    time_lag_active
    action
    -unknown-_count
    click_count
    data_count
    message_post_count
    submit_count
    view_count
    -unknown-_sum
    booking_request_sum
    booking_response_sum
    click_sum
    data_sum
    message_post_sum
    partner_callback_sum
    submit_sum
    view_sum
    -unknown-_pct
    booking_request_pct
    click_pct
    data_pct
    message_post_pct
    submit_pct
    view_pct
    action_detail



```python
categorical_features_1 = [val for val in categorical_features if val in colo]
users_encode = pd.get_dummies(users[colo], columns=categorical_features_1)
users_encode.to_csv('../cache/users_data_feature.csv')
```

### Countries and Age Bkdts dataset

Though the countries and the age dataset provide some valuable information on the destination countries and the age group of the population in these countries we find no use for them on the task at hand. If we had further information on the users in our dataset such as exact location in the US and other demographic details of users such as marital status, household income etc we could have possible used the data from the data sets to build a more robust prediction model. 


```python
from time import time
from math import sqrt
import logging
import os
import sys
import csv
import datetime


total = {}
started = {}
model_perf={}

        
def start(key):
    started[key]=time()


def stop(key):
    stop=time()
    start=started.pop(key,None)
    if start:
        if key in total:
            total[key].append(stop-float(start))
        else:
            total[key]=[stop-float(start)]
    else:
        logging.error("stopping non started timer: %s"%key)
```

## 4. Model Building

We now proceed to model building stage . But before we start training the models we perform one last data preparation  step.

We split the training data in a 80:20 ratio and use the 80 percent of the data to build the model and 20 percent of the data to validate it. While splitting we used the stratify option in the sklearn library to ensure that the data that is selected includes a representative set of the user data with the country destinations in the same proportion as the full dataset. This is very essential since the distribution of the country destination is highly unbalanced .

### Time lag variables

 The time lage variables both __time_lag_create__ and __time_lag_active__ were identified as key features both by the variance threshold method and the Decision Tree and Random forest models we built. The models built with these features included had a accuracy value of __0.88__ and nDCG score of __0.92__ . However we were not able to use these models on the test dataset since the test data set doesnt have the date of booking for the users. We do believe that these values can be used very usefully in future where we have more data points and time related informtion on the session and user data sets. For current report we excluded these two variables from models.


```python
from sklearn.model_selection import train_test_split
users = users_encode
users.set_index('id',inplace=True)
users.drop([col for col in users.columns if 'pct_booking_request' in col],axis=1,inplace=True)
users.drop([col for col in users.columns if 'booking_request_count' in col],axis=1,inplace=True)
colx = users.columns.tolist()
#colx_1 = users_data_1.columns.tolist()
rm_list = ['country_destination','month_first_book', 'time_lag_create','time_lag_active']
for x in rm_list:
    colx.remove(x)
X_1 = users[(users['country_destination'].isnull())][colx]
X_1.fillna(0,inplace=True)
X = users[~(users['country_destination'].isnull())][colx]
#X_1 = users_data_1[~(users_data_1['country_destination'].isnull())][colx_1]
Y = users[~(users['country_destination'].isnull())]['country_destination']
#Y_1 = users_data_1[~(users_data_1['country_destination'].isnull())]['country_destination']
X.fillna(0,inplace=True)
#X_1.fillna(0,inplace=True)
#X_res,Y_res = ada.fit_sample(X, Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42,stratify=Y)
#X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X_1, Y_1, test_size=0.2, random_state=42,stratify=Y)
```

## Logistic Regression - OVR
We first built a Logistic regression model optimised for OVR (One Vs Rest). Since more than 3/4 of the country destination is accounted for by 'NDF' and 'US' we evaluated the performance of the Logistic Regression model  optimised to minimize the loss for OVR.


```python
from sklearn.linear_model import LogisticRegression
start('logr')
logr = LogisticRegression()
logr.fit(X_train, Y_train)
stop('logr')
```

## Logistic Regression - Multinomial
Next we built built a Logistic regression Multinomial classification model. Since the primary objective of the problem at hand was to predict the possible destination a user will book we hypothesized that this model may have a lower accuracy score but it will do better on the nDCG score.


```python
from sklearn.linear_model import LogisticRegression
start('logr_mlt')
logr_mlt = LogisticRegression(n_jobs=1,multi_class='multinomial',solver='newton-cg')
logr_mlt.fit(X_train, Y_train)
stop('logr_mlt')
```

## SVM Linear
Next we built built a SVM-Linear Classifier model. SVM - Linear classification is not well suited to the problem at hand since they cannot predict probabilities for the target classes. However we wanted to evaluate how well the model does for a single value prediction .


```python
from sklearn.svm import LinearSVC
start('svc')
svc = LinearSVC(random_state=42)
svc.fit(X_train, Y_train)
stop('svc')
```

## Decision Tree

Next we built a decision tree classifier. Since this was a multiclass classification problem decision tree is well suited for the problem at hand . The key features in the model and their weights are detailed below.


```python
from sklearn.tree import DecisionTreeClassifier
start('dt')
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
stop('dt')
```


```python
feature_imp = pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), dt.feature_importances_), X.columns.tolist()), 
             reverse=True))
feature_imp.columns = ['value','feature']
feature_imp.set_index('feature',inplace=True)
feature_imp.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>value</th>
    </tr>
    <tr>
      <th>feature</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age_group_0</th>
      <td>0.0655</td>
    </tr>
    <tr>
      <th>data_sum</th>
      <td>0.0139</td>
    </tr>
    <tr>
      <th>first_browser_Chrome</th>
      <td>0.0130</td>
    </tr>
    <tr>
      <th>data_pct</th>
      <td>0.0130</td>
    </tr>
    <tr>
      <th>booking_request_pct</th>
      <td>0.0130</td>
    </tr>
    <tr>
      <th>view_pct</th>
      <td>0.0129</td>
    </tr>
    <tr>
      <th>click_pct</th>
      <td>0.0126</td>
    </tr>
    <tr>
      <th>message_post_count</th>
      <td>0.0124</td>
    </tr>
    <tr>
      <th>hour_first_active_22</th>
      <td>0.0123</td>
    </tr>
    <tr>
      <th>hour_first_active_19</th>
      <td>0.0121</td>
    </tr>
  </tbody>
</table>
</div>




```python
%matplotlib inline
feature_imp.nlargest(10,'value').plot(kind='barh')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc6cb736e10>




![png](Venkatesan_Karthick_Final_Project_Report_files/Venkatesan_Karthick_Final_Project_Report_113_1.png)


## Random Forest

Next we built a Random forest classifier. Random forest or random decision forest is an ensemble learning method for classification, regression and other tasks, that operate's by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forest correct for decision trees' habit of overfitting to their training set. The key features and the weights as identified by the Random Forest model are detailed below.


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
start('rf')
clf = RandomForestClassifier(n_estimators=64,n_jobs=-1)
clf.fit(X_train, Y_train)
stop('rf')
```


```python
feature_imp = pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), X.columns.tolist()), 
             reverse=True))
feature_imp.columns = ['value','feature']
feature_imp.set_index('feature',inplace=True)
feature_imp.nlargest(10,'value')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>value</th>
    </tr>
    <tr>
      <th>feature</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age_group_0</th>
      <td>0.0388</td>
    </tr>
    <tr>
      <th>gender_FEMALE</th>
      <td>0.0143</td>
    </tr>
    <tr>
      <th>first_browser_Chrome</th>
      <td>0.0137</td>
    </tr>
    <tr>
      <th>gender_MALE</th>
      <td>0.0135</td>
    </tr>
    <tr>
      <th>hour_first_active_21</th>
      <td>0.0124</td>
    </tr>
    <tr>
      <th>hour_first_active_19</th>
      <td>0.0124</td>
    </tr>
    <tr>
      <th>hour_first_active_18</th>
      <td>0.0124</td>
    </tr>
    <tr>
      <th>hour_first_active_20</th>
      <td>0.0122</td>
    </tr>
    <tr>
      <th>first_affiliate_tracked_untracked</th>
      <td>0.0122</td>
    </tr>
    <tr>
      <th>age_group_8</th>
      <td>0.0122</td>
    </tr>
  </tbody>
</table>
</div>




```python
%matplotlib inline
feature_imp.nlargest(10,'value').plot(kind='barh')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc65cdbca20>




![png](Venkatesan_Karthick_Final_Project_Report_files/Venkatesan_Karthick_Final_Project_Report_117_1.png)


## 5. Results

### Logistic Regression - OVR

The confusion matrix for the Logistic Regression - OVR method is as below. As we can see in the table the model does a pretty could job of predicting NDF however prediction accruacy is very low for all the other destinations. The accuracy for this model is __0.59__ . 

We also calculated the nDCG score by predicting the probability for each for destinations for each of the users. The Logistic Regression - OVR method got a nDCG score of __0.7618__


```python
start('logr')
y_pred_logr=logr.predict(X_test)
stop('logr')
pd.crosstab(Y_test, y_pred_logr, rownames=['Actual Destination'], colnames=['Predicted Destination'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted Destination</th>
      <th>ES</th>
      <th>NDF</th>
      <th>NL</th>
      <th>US</th>
      <th>other</th>
    </tr>
    <tr>
      <th>Actual Destination</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AU</th>
      <td>0</td>
      <td>105</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>CA</th>
      <td>0</td>
      <td>276</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>DE</th>
      <td>0</td>
      <td>203</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ES</th>
      <td>0</td>
      <td>432</td>
      <td>0</td>
      <td>18</td>
      <td>0</td>
    </tr>
    <tr>
      <th>FR</th>
      <td>3</td>
      <td>965</td>
      <td>0</td>
      <td>37</td>
      <td>0</td>
    </tr>
    <tr>
      <th>GB</th>
      <td>0</td>
      <td>440</td>
      <td>0</td>
      <td>25</td>
      <td>0</td>
    </tr>
    <tr>
      <th>IT</th>
      <td>0</td>
      <td>534</td>
      <td>0</td>
      <td>33</td>
      <td>0</td>
    </tr>
    <tr>
      <th>NDF</th>
      <td>9</td>
      <td>24540</td>
      <td>2</td>
      <td>358</td>
      <td>0</td>
    </tr>
    <tr>
      <th>NL</th>
      <td>0</td>
      <td>147</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>PT</th>
      <td>0</td>
      <td>43</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>US</th>
      <td>9</td>
      <td>11798</td>
      <td>0</td>
      <td>667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>other</th>
      <td>1</td>
      <td>1922</td>
      <td>0</td>
      <td>96</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.metrics import accuracy_score,confusion_matrix
print ('Accuracy:' + str(accuracy_score(Y_test, y_pred_logr)))
model_perf['logr','Accuracy'] = accuracy_score(Y_test, y_pred_logr)
```

    Accuracy:0.590452320161



```python
"""Metrics to compute the model performance."""

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer


def dcg_score(y_true, y_score, k=5):
    """Discounted cumulative gain (DCG) at rank K.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array, shape = [n_samples, n_classes]
        Predicted scores.
    k : int
        Rank.

    Returns
    -------
    score : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def ndcg_score(ground_truth, predictions, k=5):
    """Normalized discounted cumulative gain (NDCG) at rank K.

    Normalized Discounted Cumulative Gain (NDCG) measures the performance of a
    recommendation system based on the graded relevance of the recommended
    entities. It varies from 0.0 to 1.0, with 1.0 representing the ideal
    ranking of the entities.

    Parameters
    ----------
    ground_truth : array, shape = [n_samples]
        Ground truth (true labels represended as integers).
    predictions : array, shape = [n_samples, n_classes]
        Predicted probabilities.
    k : int
        Rank.

    Returns
    -------
    score : float

    Example
    -------
    >>> ground_truth = [1, 0, 2]
    >>> predictions = [[0.15, 0.55, 0.2], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    >>> score = ndcg_score(ground_truth, predictions, k=2)
    1.0
    >>> predictions = [[0.9, 0.5, 0.8], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    >>> score = ndcg_score(ground_truth, predictions, k=2)
    0.6666666666
    """
    lb = LabelBinarizer()
    lb.fit(range(predictions.shape[1]+ 1))
    T = lb.transform(ground_truth)


    scores = []

    # Iterate over each y_true and compute the DCG score
    for y_true, y_score in zip(T, predictions):
        actual = dcg_score(y_true, y_score, k)
        best = dcg_score(y_true, y_true, k)
        score = float(actual) / float(best)
        scores.append(score)

    return np.mean(scores)

y_conv = [clf.classes_.tolist().index(k) for k in Y_test.tolist()] 
start('logr')
y_pred_prob_logr=logr.predict_proba(X_test)
stop('logr')
print('nDCG:' + str(ndcg_score(y_conv,y_pred_prob_logr)))
model_perf['logr','nDCG'] = ndcg_score(y_conv,y_pred_prob_logr)
#pd.DataFrame(y_pred_prob,columns=clf.classes_)
```

    nDCG:0.761802605855


### Logistic Regression - Multinomial

The confusion matrix for the Logistic Regression - Multinomial method is as below. As we can see in the table the model does a pretty could job of predicting NDF however prediction accruacy is very low for all the other destinations. The accuracy for this model is __0.58__ . 

We also calculated the nDCG score by predicting the probability for each for destinations for each of the users. The Logistic Regression - Multinomial method got a nDCG score of __0.8070__.

Also as we had hypothesized though this model was less accurate than the Logistic regression - OVR model we got a significantly higher nDCG score than the Logistic regression - OVR model.


```python
start('logr_mlt')
y_pred_logr_mlt=logr_mlt.predict(X_test)
stop('logr_mlt')
pd.crosstab(Y_test, y_pred_logr_mlt, rownames=['Actual Destination'], colnames=['Predicted Destination'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted Destination</th>
      <th>NDF</th>
      <th>PT</th>
      <th>US</th>
    </tr>
    <tr>
      <th>Actual Destination</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AU</th>
      <td>107</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CA</th>
      <td>282</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>DE</th>
      <td>206</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>ES</th>
      <td>440</td>
      <td>0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>FR</th>
      <td>989</td>
      <td>0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>GB</th>
      <td>448</td>
      <td>0</td>
      <td>17</td>
    </tr>
    <tr>
      <th>IT</th>
      <td>547</td>
      <td>0</td>
      <td>20</td>
    </tr>
    <tr>
      <th>NDF</th>
      <td>24675</td>
      <td>1</td>
      <td>233</td>
    </tr>
    <tr>
      <th>NL</th>
      <td>148</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>PT</th>
      <td>42</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>US</th>
      <td>12082</td>
      <td>0</td>
      <td>393</td>
    </tr>
    <tr>
      <th>other</th>
      <td>1951</td>
      <td>0</td>
      <td>68</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.metrics import accuracy_score,confusion_matrix
print ('Accuracy:' + str(accuracy_score(Y_test, y_pred_logr_mlt)))
model_perf['logr_mlt','Accuracy'] = accuracy_score(Y_test, y_pred_logr_mlt)
```

    Accuracy:0.587196364573



```python
start('logr_mlt')
y_pred_prob_logr_mlt=logr_mlt.predict_proba(X_test)
stop('logr_mlt')
print('nDCG:' + str(ndcg_score(y_conv,y_pred_prob_logr_mlt)))
model_perf['logr_mlt','nDCG'] = ndcg_score(y_conv,y_pred_prob_logr_mlt)
```

    nDCG:0.807007117608


### SVM Linear
The confusion matrix for the SVM Linear Classifier method is as below. As we can see in the table the model fares very badly in predicting almost all destinations. The accuracy for this model is __0.48__ . This indicates the decision bouandries are not linear and we should explore non linear SVM model. We were unable to compute the nDCG score for this model as the SVM Linear 
method cannot predict probabilities for all the classes.


```python
start('svc')
y_pred_svc=svc.predict(X_test)
stop('svc')
pd.crosstab(Y_test, y_pred_svc, rownames=['Actual Destination'], colnames=['Predicted Destination'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted Destination</th>
      <th>AU</th>
      <th>CA</th>
      <th>DE</th>
      <th>ES</th>
      <th>FR</th>
      <th>GB</th>
      <th>IT</th>
      <th>NDF</th>
      <th>NL</th>
      <th>US</th>
      <th>other</th>
    </tr>
    <tr>
      <th>Actual Destination</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AU</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>1</td>
      <td>62</td>
      <td>0</td>
      <td>35</td>
      <td>0</td>
    </tr>
    <tr>
      <th>CA</th>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>26</td>
      <td>0</td>
      <td>1</td>
      <td>159</td>
      <td>1</td>
      <td>87</td>
      <td>2</td>
    </tr>
    <tr>
      <th>DE</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>13</td>
      <td>0</td>
      <td>1</td>
      <td>129</td>
      <td>0</td>
      <td>62</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ES</th>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>46</td>
      <td>0</td>
      <td>3</td>
      <td>252</td>
      <td>1</td>
      <td>141</td>
      <td>0</td>
    </tr>
    <tr>
      <th>FR</th>
      <td>4</td>
      <td>19</td>
      <td>0</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>3</td>
      <td>568</td>
      <td>5</td>
      <td>323</td>
      <td>1</td>
    </tr>
    <tr>
      <th>GB</th>
      <td>4</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>47</td>
      <td>0</td>
      <td>5</td>
      <td>253</td>
      <td>0</td>
      <td>142</td>
      <td>2</td>
    </tr>
    <tr>
      <th>IT</th>
      <td>5</td>
      <td>11</td>
      <td>1</td>
      <td>0</td>
      <td>60</td>
      <td>1</td>
      <td>4</td>
      <td>310</td>
      <td>3</td>
      <td>169</td>
      <td>3</td>
    </tr>
    <tr>
      <th>NDF</th>
      <td>119</td>
      <td>544</td>
      <td>38</td>
      <td>116</td>
      <td>3028</td>
      <td>26</td>
      <td>120</td>
      <td>16546</td>
      <td>69</td>
      <td>4250</td>
      <td>53</td>
    </tr>
    <tr>
      <th>NL</th>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>0</td>
      <td>1</td>
      <td>77</td>
      <td>0</td>
      <td>50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>PT</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>18</td>
      <td>0</td>
    </tr>
    <tr>
      <th>US</th>
      <td>58</td>
      <td>249</td>
      <td>14</td>
      <td>29</td>
      <td>1084</td>
      <td>10</td>
      <td>79</td>
      <td>6925</td>
      <td>20</td>
      <td>3961</td>
      <td>46</td>
    </tr>
    <tr>
      <th>other</th>
      <td>17</td>
      <td>51</td>
      <td>5</td>
      <td>4</td>
      <td>219</td>
      <td>1</td>
      <td>10</td>
      <td>1096</td>
      <td>6</td>
      <td>603</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.metrics import accuracy_score,confusion_matrix
print ('Accuracy:' + str(accuracy_score(Y_test, y_pred_svc)))
model_perf['svc','Accuracy'] = accuracy_score(Y_test, y_pred_svc)
```

    Accuracy:0.482724696072


### Decision Tree

The confusion matrix for the Decision tree classifier is as below. As we can see in the table the model does a pretty could job of predicting NDF however predictiong is very low for all the other destinations. The accuracy for this model is __0.51__. 

We also calculated the nDCG score by predicting the probability for each for destinations for each of the users. The Decision tree model got a nDCG score of __0.69__


```python
start('dt')
y_pred_dt=dt.predict(X_test)
stop('dt')
pd.crosstab(Y_test, y_pred_dt, rownames=['Actual Destination'], colnames=['Predicted Destination'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted Destination</th>
      <th>AU</th>
      <th>CA</th>
      <th>DE</th>
      <th>ES</th>
      <th>FR</th>
      <th>GB</th>
      <th>IT</th>
      <th>NDF</th>
      <th>NL</th>
      <th>PT</th>
      <th>US</th>
      <th>other</th>
    </tr>
    <tr>
      <th>Actual Destination</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AU</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>2</td>
      <td>0</td>
      <td>37</td>
      <td>7</td>
    </tr>
    <tr>
      <th>CA</th>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>11</td>
      <td>3</td>
      <td>5</td>
      <td>137</td>
      <td>2</td>
      <td>0</td>
      <td>106</td>
      <td>15</td>
    </tr>
    <tr>
      <th>DE</th>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>90</td>
      <td>0</td>
      <td>0</td>
      <td>83</td>
      <td>14</td>
    </tr>
    <tr>
      <th>ES</th>
      <td>6</td>
      <td>3</td>
      <td>2</td>
      <td>7</td>
      <td>16</td>
      <td>6</td>
      <td>4</td>
      <td>209</td>
      <td>2</td>
      <td>1</td>
      <td>173</td>
      <td>21</td>
    </tr>
    <tr>
      <th>FR</th>
      <td>4</td>
      <td>6</td>
      <td>6</td>
      <td>18</td>
      <td>32</td>
      <td>16</td>
      <td>17</td>
      <td>505</td>
      <td>4</td>
      <td>0</td>
      <td>351</td>
      <td>46</td>
    </tr>
    <tr>
      <th>GB</th>
      <td>0</td>
      <td>3</td>
      <td>6</td>
      <td>12</td>
      <td>7</td>
      <td>1</td>
      <td>11</td>
      <td>234</td>
      <td>2</td>
      <td>0</td>
      <td>159</td>
      <td>30</td>
    </tr>
    <tr>
      <th>IT</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>17</td>
      <td>8</td>
      <td>9</td>
      <td>292</td>
      <td>1</td>
      <td>1</td>
      <td>203</td>
      <td>24</td>
    </tr>
    <tr>
      <th>NDF</th>
      <td>45</td>
      <td>153</td>
      <td>91</td>
      <td>226</td>
      <td>453</td>
      <td>220</td>
      <td>304</td>
      <td>16925</td>
      <td>61</td>
      <td>16</td>
      <td>5559</td>
      <td>856</td>
    </tr>
    <tr>
      <th>NL</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>4</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>67</td>
      <td>8</td>
    </tr>
    <tr>
      <th>PT</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>22</td>
      <td>1</td>
      <td>0</td>
      <td>16</td>
      <td>1</td>
    </tr>
    <tr>
      <th>US</th>
      <td>33</td>
      <td>109</td>
      <td>72</td>
      <td>156</td>
      <td>383</td>
      <td>169</td>
      <td>201</td>
      <td>5869</td>
      <td>46</td>
      <td>17</td>
      <td>4705</td>
      <td>715</td>
    </tr>
    <tr>
      <th>other</th>
      <td>6</td>
      <td>17</td>
      <td>8</td>
      <td>30</td>
      <td>53</td>
      <td>34</td>
      <td>39</td>
      <td>995</td>
      <td>6</td>
      <td>4</td>
      <td>702</td>
      <td>125</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.metrics import accuracy_score,confusion_matrix
print ('Accuracy:' + str(accuracy_score(Y_test, y_pred_dt)))
model_perf['dt','Accuracy'] = accuracy_score(Y_test, y_pred_dt)
```

    Accuracy:0.510786816893



```python
start('dt')
y_pred_prob_dt=dt.predict_proba(X_test)
stop('dt')
print('nDCG:' + str(ndcg_score(y_conv,y_pred_prob_dt)))
ndcg_score(y_conv,y_pred_prob_dt)
model_perf['dt','nDCG'] = ndcg_score(y_conv,y_pred_prob_dt)
```

    nDCG:0.692985103317


###  Random Forest

The confusion matrix for the Random tree classifier is as below. As we can see in the table the model does a pretty could job of predicting NDF as well US destinations however accuracy is very low for all the other destinations. The accuracy for this model is __0.60__ . 

We also calculated the nDCG score by predicting the probability for each for destinations for each of the users. The Decion tree model got a nDCG score of __0.80__.


```python
start('rf')
y_pred=clf.predict(X_test)
stop('rf')
pd.crosstab(Y_test, y_pred, rownames=['Actual Destination'], colnames=['Predicted Destination'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted Destination</th>
      <th>AU</th>
      <th>CA</th>
      <th>DE</th>
      <th>ES</th>
      <th>FR</th>
      <th>GB</th>
      <th>IT</th>
      <th>NDF</th>
      <th>NL</th>
      <th>PT</th>
      <th>US</th>
      <th>other</th>
    </tr>
    <tr>
      <th>Actual Destination</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AU</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>65</td>
      <td>0</td>
      <td>0</td>
      <td>42</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CA</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>164</td>
      <td>0</td>
      <td>0</td>
      <td>117</td>
      <td>3</td>
    </tr>
    <tr>
      <th>DE</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>129</td>
      <td>0</td>
      <td>0</td>
      <td>81</td>
      <td>2</td>
    </tr>
    <tr>
      <th>ES</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>278</td>
      <td>0</td>
      <td>0</td>
      <td>164</td>
      <td>2</td>
    </tr>
    <tr>
      <th>FR</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>609</td>
      <td>1</td>
      <td>0</td>
      <td>388</td>
      <td>5</td>
    </tr>
    <tr>
      <th>GB</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>285</td>
      <td>1</td>
      <td>0</td>
      <td>174</td>
      <td>3</td>
    </tr>
    <tr>
      <th>IT</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>347</td>
      <td>0</td>
      <td>0</td>
      <td>216</td>
      <td>0</td>
    </tr>
    <tr>
      <th>NDF</th>
      <td>3</td>
      <td>13</td>
      <td>8</td>
      <td>14</td>
      <td>46</td>
      <td>19</td>
      <td>32</td>
      <td>20741</td>
      <td>3</td>
      <td>1</td>
      <td>3926</td>
      <td>103</td>
    </tr>
    <tr>
      <th>NL</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>92</td>
      <td>0</td>
      <td>0</td>
      <td>58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>PT</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>US</th>
      <td>0</td>
      <td>4</td>
      <td>7</td>
      <td>9</td>
      <td>28</td>
      <td>6</td>
      <td>10</td>
      <td>7315</td>
      <td>5</td>
      <td>0</td>
      <td>5034</td>
      <td>57</td>
    </tr>
    <tr>
      <th>other</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>1226</td>
      <td>0</td>
      <td>0</td>
      <td>769</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.metrics import accuracy_score,confusion_matrix
print ('Accuracy:' + str(accuracy_score(Y_test, y_pred)))
model_perf['rf','Accuracy'] = accuracy_score(Y_test, y_pred)
```

    Accuracy:0.604038321895



```python
start('rf')
y_pred_prob=clf.predict_proba(X_test)
stop('rf')
print('nDCG:' + str(ndcg_score(y_conv,y_pred_prob)))
model_perf['rf','nDCG'] = ndcg_score(y_conv,y_pred_prob)
```

    nDCG:0.805144161362


### Predict for Test Dataset

As a final step we predict the top 5 destination countries for each users based on the probabilities predicted from the trained models and order the results in the format as mandated by the competitions and submit it to kaggle. Below is a screen shot of submissions from Kaggle. As depicted in the summary table the Random Forest Model performed best of all the models closely followed by Logistic Regression Multinomial. 

Also from the random forest method did excellently on speed as well and its run times for the same data was only 12 seconds compared to nearly 700+ seconds run time of the Logistic regression multinomial which was the closest in terms on the nDCG score in our final submission.


```python
y_pred_prob_1=dt.predict_proba(X_1)
id_test = X_1.reset_index()['id']
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    arr = [clf.classes_.tolist()[k] for k in np.argsort(y_pred_prob_1[i])[::-1]] 
    cts += arr[:5]

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('../data/sub_dt.csv',index=False)
```


```python
y_pred_prob_1=clf.predict_proba(X_1)
id_test = X_1.reset_index()['id']
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    arr = [clf.classes_.tolist()[k] for k in np.argsort(y_pred_prob_1[i])[::-1]] 
    cts += arr[:5]

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('../data/sub_rf.csv',index=False)
```


```python
y_pred_prob_1=logr.predict_proba(X_1)
id_test = X_1.reset_index()['id']
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    arr = [clf.classes_.tolist()[k] for k in np.argsort(y_pred_prob_1[i])[::-1]] 
    cts += arr[:5]

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('../data/sub_logr.csv',index=False)
```


```python
y_pred_prob_1=logr_mlt.predict_proba(X_1)
id_test = X_1.reset_index()['id']
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    arr = [clf.classes_.tolist()[k] for k in np.argsort(y_pred_prob_1[i])[::-1]] 
    cts += arr[:5]

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('../data/sub_logr_mlt.csv',index=False)
```

#### Sample Output


```python
sub.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5uwns89zht</td>
      <td>NDF</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5uwns89zht</td>
      <td>US</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5uwns89zht</td>
      <td>other</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5uwns89zht</td>
      <td>FR</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5uwns89zht</td>
      <td>IT</td>
    </tr>
    <tr>
      <th>5</th>
      <td>jtl0dijy2j</td>
      <td>NDF</td>
    </tr>
    <tr>
      <th>6</th>
      <td>jtl0dijy2j</td>
      <td>US</td>
    </tr>
    <tr>
      <th>7</th>
      <td>jtl0dijy2j</td>
      <td>other</td>
    </tr>
    <tr>
      <th>8</th>
      <td>jtl0dijy2j</td>
      <td>FR</td>
    </tr>
    <tr>
      <th>9</th>
      <td>jtl0dijy2j</td>
      <td>IT</td>
    </tr>
  </tbody>
</table>
</div>



#### Kaggle Results


```python
from IPython.display import Image
Image("../data/Kaggle.png")
```




![png](Venkatesan_Karthick_Final_Project_Report_files/Venkatesan_Karthick_Final_Project_Report_148_0.png)



#### Summary Results


```python
model1 = {}
value = {}
kag= {'dt':0.75291,'logr':0.84852,'logr_mlt':0.85352,'rf':0.86449}
desc = {'dt':'Decision Tree','logr':'Logistic Regression OVR','logr_mlt':'Logistic Regression - Multinomial','rf':'Random Forest','svc':'Support Vector Classifier'}
for j in model_perf.keys():
    if desc[j[0]] not in model1.keys():
        value = {}
        value['Time(s)'] = total[j[0]][0]
        if j[0] in kag.keys():
            value['kaggle_nDCG'] = kag[j[0]] 
    value[j[1]] = model_perf[j]
    model1[desc[j[0]]] = value
        
pd.DataFrame(model1).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Time(s)</th>
      <th>kaggle_nDCG</th>
      <th>nDCG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Decision Tree</th>
      <td>0.510787</td>
      <td>7.964606</td>
      <td>0.75291</td>
      <td>0.692985</td>
    </tr>
    <tr>
      <th>Logistic Regression - Multinomial</th>
      <td>0.587196</td>
      <td>714.866235</td>
      <td>0.85352</td>
      <td>0.807007</td>
    </tr>
    <tr>
      <th>Logistic Regression OVR</th>
      <td>0.590452</td>
      <td>251.514795</td>
      <td>0.84852</td>
      <td>0.761803</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.604038</td>
      <td>11.925010</td>
      <td>0.86449</td>
      <td>0.805144</td>
    </tr>
    <tr>
      <th>Support Vector Classifier</th>
      <td>0.482725</td>
      <td>467.602295</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## 6. Conclusion

In conclusion we can see that the models based on Random Forest and Logistic Regression OVR , Logistic Regression Multinomial and Decision trees were able to predict the destination country of the users with more than 50 % accuracy and nDCG score greater than 0.75. The variables
identified as important in the model included both demographic features, such as gender, age, and
language, as well as the features we generated using the web sessions data, such as sum of
elapsed seconds and the count of unique user actions. Overall, we can see that the random
forest model is the best for the purpose of predicting whether a user will make a booking, as well as the destinations
of these bookings. In addition to be being the most accurate model the Random Forest model also was the fastest interms of runtimes compared to the other models. 

In future, we could further extend the study by building models such as Grandient Bosted trees and Neural networks. These models will require significant resources and most likely improve on the accuracy rates. We could also explore an ensemble classifer by using a combination of models for example both Logistic Regression and Random Forest. We could use Logistic Regression to perform the binary classification, and random forest to perform the multinomial classification. This will help us to better map the decision boundaries in the data set. Also as noted in the results age and geneder were significant in predicting the destination country . We could build prediction models and predict these missing demographic values based on the data in the sessions data set which would also help to improve the accuracy of our models.

The following additional data if present will significantly help in improving the prediction accuracy of the models. Firstly for almost 40% of the users we are missing the session data , this data if present will help in building better prediction models. Secondly the session data doesnt have a date for when the action was done. If we had this data we can use it to more accurately predict the destination booking by calculating the time lag variables which as already noted in the study were excellent predictors of country destination. Additionally, the default demographic data in Airbnb’s is fairly limited, and the predictors that
are included are littered with missing values and human input errors. One key predictor that could
potentially improve our predictions is the location of a user in the United States. For example, certain
cities in the United States may have sizable immigrant populations with ties back to European countries.
Residence in a particular city or region may suggest a higher likelihood of making a booking in a
particular. country. We also lack information about the potential number of travellers for a single
booking. Knowing that a user has a spouse and two kids in the household may suggest a higher
likelihood of booking in different places.


```python

```
