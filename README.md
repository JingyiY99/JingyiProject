# Nanocellulose Enhanced Fish Gel


## Problem Statement
Asian Carps are invasive species in the Great Lakes and the Mississippi River systems, and they have become a threat to native fishes, creating many serious ecological problem. We should find a way to increase the usability of Asian Carps as food resourses. Making Surimi food is a great solution. Surimi, a kind of protein-rich food made by fish mince, is gaining popularity in the U.S and Europe, and already very popular in Asian countries.  
  
![Book logo](/JingyiProject/assets/Asian carps.png)  
Asian carps BY TIFFANY JOLLEY, blogs.illinois.edu  
  
Most surimi gel is made by fresh fish because the frozen-stored process will degrade the proteins that impact the cross-linking property. In general, surimi gels made from frozen stored fishes tend to have decreased water holding capacity and poor gel texture and strength. However, using fresh fish will increase the cost of surimi prduction.  
  
![Book logo](/JingyiProject/assets/wx_camera_1621624272690.jpg)
Surimi gel we made  
  
Finding innovative ways to produce high-quality surimi from frozen-stored Asian carps will use an under-utilized food source to create high-value products and provide a strong economic incentive to reduce these invasive species and lessen their detrimental environmental, ecological impacts on the great lakes and Mississippi River systems.  
There is a methods to increase the protein's cross-linking ability: Nanocellulose (NC). NC is a good gel enhancer that makes network into proteins to help cross-link and increase the water holding capacity. NC also has some health benefits like improving bowel movement, promoting healthy microbiota in digestive tract, and trapping heavy metals out of harm’s way. 
  
![Book logo](/JingyiProject/assets/H2aa39567e41e40298725ebd72b9b69535.png)  
Nanocellulose, alibaba.com  
  
We should find whether the NC enhnced gel has better gel quality and the relationship between the dosage of NC and gel quality.  

## Data Description
I make some gel samples with different NC dasage: no NC, 0.1%wt NC， 0.5%wt NC, and 1.0%wt NC. Then we need to collect data to compare their gel quality.  
Texture Profile Analysis (TPA) test is useful to measure the gel properties like hardness, cohesiveness, springiness, chewiness, resilience, and gel strength. To determine the gel quality of our samples, we will focus on hardness and gel strength, but other properties can help us find the relationship between NC dasage and gel properties.  
The data I use to analyze is measured on November 12th 2021 by TPA machine in Food Science Building. You can find it <a href="https://github.com/JingyiY99/JingyiProject/blob/master/assets/Result_11.12.csv" download>here</a>.  

## Explore the Data
First, let's take a look at our <a href="https://github.com/JingyiY99/JingyiProject/blob/master/assets/Result_11.12.csv" download>data</a>. Make sure there is no negative valuses and NA values. 
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
  
df = pd.read_csv('Result_11.12.csv')
df = df.drop(columns=["Distance1","Distance2","Area1","Area2","Area3","Area4"])
df
```
![Book logo](/JingyiProject/assets/Raw data.PNG)  
Then we can calculate the means and standard deviations for each dosage to describe the basic statistics of the features.
```python
class0 = df[df["Dosage"]==0]
class0 = class0.drop(columns=["Dosage"])
sum0 = class0.describe()
sum0
  
class1 = df[df["Dosage"]==0.1]
class1 = class1.drop(columns=["Dosage"])
sum1 = class1.describe()
sum1
  
class2 = df[df["Dosage"]==0.5]
class2 = class2.drop(columns=["Dosage"])
sum2 = class2.describe()
sum2
  
class3 = df[df["Dosage"]==1]
class3 = class3.drop(columns=["Dosage"])
sum3 = class3.describe()
sum3
```
Results for no NC enhanced gel:  
![Book logo](/JingyiProject/assets/no NC sum.PNG)  
Results for 0.1% NC enhanced gel:  
![alt text](/JingyiProject/assets/0.1NC.PNG)  
Results for 0.5% NC enhanced gel:  
![alt text](/JingyiProject/assets/0.5NC.PNG)  
Results for 1.0% NC enhanced gel:  
![alt text](/JingyiProject/assets/1.0NC.PNG)  
There is no obvious issue based on the preliminary summary. Let's dig deeper.  

## Data Visualization
Boxplot is a good visualization method to show the replationships between NC dosage and each features.
```python
def features(x): 
 """"" 
 Find the features by x. 
 """""
 if x == 0: 
    return "Hardness" 
 elif x == 1: 
    return "Cohesiveness"
 elif x == 2: 
    return "Springiness"
 elif x == 3: 
    return "Gumminess"
 elif x == 4: 
    return "Chewiness"
 elif x == 5: 
    return "Resilience"
 elif x == 6: 
    return "Gel Strength"
    
x = np.arange(0,7) 
for n in x: 
    data = [class0[features(n)],class1[features(n)], class2[features(n)], class3[features(n)]] 
    fig = plt.figure(figsize =(10, 7)) 
    ax = fig.add_subplot(111) 
    bp = ax.boxplot(data) 
    plt.title('Dosage(%wt) vs. '+ features(n))
    ax.set_xticklabels(['0%wt','0.1%wt', '0.5%wt', '1.0%wt']) 
    plt.show(bp)
```
![alt text](/JingyiProject/assets/hardness.PNG)  
![alt text](/JingyiProject/assets/Cohesiveness.PNG)  
![alt text](/JingyiProject/assets/Springiness.PNG)  
![alt text](/JingyiProject/assets/Gumminess.PNG)  
![alt text](/JingyiProject/assets/Chewiness.PNG)  
![alt text](/JingyiProject/assets/Resilience.PNG)  
![alt text](/JingyiProject/assets/Gel Strength.PNG)  
From the boxplots, we can compare the features of difeerent NC dosage. The hardness and gel strength graphs show that no NC and 0.5% NC enhanced gel has better gel quality. Then we can compare the other features and find that 0.5% NC enhanced gel perform better on other features than no NC enhanced gel does. Therefore, we can conclude that NC enhanced gel can increase the gel quality and 0.5% NC dosage performs the better results.  

## Model the data (Machine Learning)
After using the data visualization method, let's use the Machine Learning method the model the data and compare the performence of these models.
### Naive Bayes
While most machine learning models try to predict values from the broad population, Bayesian Statistics can help us determine whether the data confirms or refutes our hypothesis more easily. It provides an easier way to get prior information from data. It can help you design your experiment with Bayesian probability. The use of Bayesian Statistics is broad that fits a wide range of the model. Naive Bayes assumes that all predictors are independent. This assumption limits the applicability of this algorithm in real-world use cases.
```python
# imports 
import pandas as pd 
import seaborn as sns 
import statsmodels.formula.api as smf 
from sklearn.linear_model import LinearRegression 
from sklearn import metrics 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
import numpy as np 
# allow plots to appear directly in the notebook 
%matplotlib inline 
  
X = np.stack((df["Hardness"], df["Cohesiveness"], df["Springiness"], df["Gumminess"], df["Chewiness"], df["Resilience"],df["Gel Strength"]), axis=1) 
Y = df["Dosage"]*10 #sklearn CAN NOT read the decimal like 0.1, so I multiply them ny 10 to get integer. Therefore, 0 is no NC, 1 is 0.1% NC, 5 is 0.5% NC, 10 is 1% NC 
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)  #Fit Gaussian Naive Bayes according to X, y.
GaussianNB()
clf.score(X, Y, sample_weight=None)
```
The performence score of Naive Bayes model is 0.6. It's larger than 0.5 but not large enough. Let's see next model.
### SVM
SVM method is broadly used in classification problems. It plots each dataset in n-dimensional space with the value of a particular coordinate. Then, it will find the hyper-plane that differentiates the classes very well. SVM is more effective in high dimensional spaces, especially where the number of dimensions is greater than the number of samples. SVM algorithm is not suitable for large data sets with more noise.   
```python
%matplotlib inline 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats 
import pandas as pd 
from pandas import DataFrame 
 
# use seaborn plotting defaults 
import seaborn as sns; sns.set()   
  
X = X = np.stack((df["Hardness"], df["Cohesiveness"], df["Springiness"], df["Gumminess"], df["Chewiness"], df["Resilience"],df["Gel Strength"]), axis=1)
y = df["Dosage"]*10
  
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) 
  
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
  
from sklearn.svm import SVC 
classifier = SVC(kernel = 'linear', C=1, random_state = 0) 
classifier.fit(X_train, Y_train) 
  
classifier.score(X_train,Y_train, sample_weight=None)
```
The performence score of SVM model is 0.909. It's good enough to fit my data.
### PCA
Naive Bayes and SVM are both supervised learning model that uses training data to learn a relationship between the input and the outputs. In an unsupervised learning model like PCA and kmeans, only input data will be given. Unsupervised learning does not use output data.   
There are too many property variables. I want to reduce the number of variables and ensure my variables are independent of one another. In this case, it's comfortable making the independent variables less interpretable. Therefore, I will use the PCA method to analyze my data.  
PCA method can reduce correlated features, reduce overfitting, and improve visualization by reducing variables. It has disadvantages that the independent variables will become less interpretable. Before using the PCA method, we should standardize the data. Otherwise, PCA will not be able to find the optimal principal components.  
```python
%matplotlib inline 
import seaborn as sns; sns.set()
  
print(df.shape)
```
The data shape is (15,8)  
```python
from sklearn.preprocessing import StandardScaler

X_data = df.iloc[:, 1:7]
Y_data = df.iloc[:,0]

scaled_data = StandardScaler()
scaled_X = scaled_data.fit_transform(X_data)
  
from sklearn.decomposition import PCA

pca1 = PCA(n_components=4)
pca1.fit(scaled_X)
train_pca1 = pca1.transform(scaled_X)  
  
pc_df = pd.DataFrame(data = train_pca1, columns = ['PC1', 'PC2','PC3','PC4'])
pc_df['Dosage'] = Y_data
pc_df
  
pca1.explained_variance_ratio_
  
df = pd.DataFrame({'var':pca1.explained_variance_ratio_, 'PC':['PC1','PC2','PC3','PC4']})
sns.barplot(x='PC',y="var", data=df, color="red");
```
![alt text](/JingyiProject/assets/PCA1.PNG)  
From the graph, it's obvious that PC1 and PC2 will be a good approximation for my data analysis.  
```python
p = sns.lmplot( x="PC1", y="PC2", data=pc_df, fit_reg=False, hue='Dosage', legend=True) 
p
```
![alt text](/JingyiProject/assets/pca2.PNG)  
From the plots, we can see the PC1 and PC2 separate the data. The PCA method successfully reduces the variables and simulates a good enough model.
## Communciate and visualize the results
We find the appropriate Nanocellulose can increase the gel quality from the Data Visualization part. Based on the boxplots, the 0.5%wt NC enhanced gel has the best performance. Both the lower dosage (0.1%wt) and higher dosage (1.0%wt) will decrease the gel quality.
The Machine Learning models are helpful to fit my data and simulate the relationship. The performence score for Naive Bayes is 0.6, for SVM is 0.909. The variance check for PCA model is good enough, too. With these models, I can predict the results for new data in the future.  
The next step for my research is to test the NC dosage around 0.5%, like 0.3~0.8%, to find a more accurate dosage.  
In this project, I applicate many topics I learned in ABE 516x. For example:  
(1)Summarizie data through python to describe the basic statistics;  
(2)Make boxplots to identify of patterns and relationships through python;  
(3)Use different machine learning tools that I learned in class to fit my data and make predictions.  
My project is easy to reproduce. You can find the data I used <a href="https://github.com/JingyiY99/JingyiProject/blob/master/assets/Result_11.12.csv" download>here</a>. The TPA test data has its global standard. Therefore, everyone can use their data measured by the TPA test to replace the data in my code. If your data has a different number of features and dosage, there would be some error, so please change the features and dosage in my code to fit your data. The other procedure is the same as I do. The data analysis method I used in the project is reproducible and easy to work through. 
## Task for Class
Water holding capacity(WHC) is also important for surimi gel production. <a href="https://github.com/JingyiY99/JingyiProject/blob/master/assets/WHC_12.2.csv" download>Here</a> is the WHC data for different NC enhanced gel. Please follow what I did in the Data Visualization part and make a boxplot to show the relationship between WHC and NC dosage. This time you have one feature with different number of dosage to analyze, so remember to change them in the code.  
## Data Workflow Chart
![Book logo](/JingyiProject/assets/Project Workflow.png)
