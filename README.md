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
I make some gel samples with different NC dasage: no NC, 0.1% NC， 0.5% NC, and 1.0% NC. Then we need to collect data to compare their gel quality.  
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
There is no obvious issue based on the preliminary summary. Let's dig depper.  

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
After using the data visualization method, let's use the Machine Learning method the model the data and compare the performence score of the models.
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

### K-means

## Communciate and visualize the results


## Data Workflow Chart
![Book logo](/JingyiProject/assets/Project Workflow-1.png)
