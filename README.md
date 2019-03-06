# Multiple Linear Regression using Scikit-Learn

This project is about Multiple Linear Regression which is a machine learning algorithm. I build a multiple linear regression 
model to estimate the relative CPU performance of computer hardware dataset.


## Table of contents

The contents of this project are divided into various categories which are given as follows:-


1.	Introduction

2.	Linear regression intuition

    * Regression line
    
    * Cost function
    
    * Ordinary least square method

3.	Independent and dependent variables

4.	Assumptions of  linear regression

5.	The dataset description

6.	The problem statement

7.	Import the Python libraries

8.	Import the dataset

9.	Exploratory Data Analysis

    * Explore types of variables
    
    * Estimate correlation coefficients
    
    * Correlation heat map
    
10.	Detect problems within variables

    * Detect missing values
    
    * Outliers in discrete variables
    
    * Number of labels – cardinality
    
11.	Linear Regression modeling

    * Divide the dataset into categorical and numerical variables
    
    *	Select the predictor and target variables
    
    * Create separate train and test sets
    
    * Feature Scaling
    
    * Fit the Linear Regression model

12.	Predicting the results

    *	Predicting the test set results

    * Predicting estimated relative CPU performance values

13.	Model slope and intercept terms

14.	Evaluate model performance

    *	RMSE (Root Mean Square Error)
    
    *	R2 Score
    
    * Overfitting or Underfitting
    
    *	Cross validation
    
    * Residual analysis
    
    * Normality test (Q-Q Plot)

15.	Conclusion

16.	Inferences



=================================================================================


## 1. Introduction

In this project, I build a multiple linear regression model to estimate the relative CPU performance of computer hardware dataset. Relative CPU performance of the computer hardware is described in terms of machine cycle time, main memory, cache memory and minimum
and maximum channels as given in the dataset.

I discuss the basics and assumptions of linear regression. I also discuss the advantages and disadvantages and common pitfalls of 
linear regression. I present the implementation in Python programming language using Scikit-learn. Scikit-learn is the popular machine learning library of Python programming language. I also discuss various tools to evaluate the linear regression model performance.



=================================================================================

## 2. Linear regression intuition

Linear Regression is a machine learning algorithm which is used to establish the linear relationship between dependent and one or more independent variables. This technique is applicable for supervised learning regression problems where we try to predict a continuous variable. Linear Regression can be further classified into two types – Simple and Multiple Linear Regression. 



### Regression line

In multiple linear regression, our task is to find a line which best fits the above scatter plot. This line will help us to predict 
the value of any Target variable for any given Feature variable. This line is called **regression line**. 

We can define an error function for any line. Then, the regression line is the one which minimizes the error function. Such an error function is also called a **Cost function**. 


### Cost Function

We want the above line to resemble the dataset as closely as possible. In other words, we want the line to be as close to actual data points as possible. It can be achieved by minimizing the vertical distance between the actual data point and fitted line. We calculate the vertical distance between each data point and the line. This distance is called the **residual**. So, in a regression model, we try to minimize the residuals by finding the line of best fit. 

We can try to minimize the sum of the residuals, but then a large positive residual would cancel out a large negative residual. For this reason, we minimize the sum of the squares of the residuals. 

Mathematically, we denote actual data points by yi and predicted data points by ŷi. So, the residual for a data point i would be given as 
				
di = yi -  ŷi

Sum of the squares of the residuals is given as:

				D = Ʃ di2       for all data points

This is the **Cost function**. It denotes the total error present in the model which is the sum of the total errors of each individual data point. 



### Ordinary Least Square Method


We can estimate the parameters of the model by minimizing the error in the model by minimizing D. Thus, we can find the 
regression line. This method of finding the parameters of the model and thus regression line is called 
**Ordinary Least Square Method**.


In this project, I employ Multiple Linear Regression technique where I have one dependent variable and more than one independent variables.


================================================================================


## 3. Independent and dependent variables

In this project, I refer Independent variable as Feature variable and Dependent variable as Target variable. These variables are 
also recognized by different names as follows: -

### Independent variable

Independent variable is also called Input variable and is denoted by X. In practical applications, independent variable is also called Feature variable or Predictor variable. We can denote it as follows: -

**Independent or Input variable (X) = Feature variable = Predictor variable** 

### Dependent variable

Dependent variable is also called Output variable and is denoted by y. Dependent variable is also called Target variable or Response variable. It can be denoted it as follows: -

**Dependent or Output variable (y) = Target variable = Response variable**


=================================================================================


## 4. Assumptions of Linear Regression


The Linear Regression model is based on several assumptions which are as follows:-


**1. Linear relationship**


The first assumption requires that **the independent variables must be linearly related to dependent variables**. As the name suggests, it maps linear relationships between dependent and independent variables.  It is also important to check for outliers since linear regression is sensitive to outlier effects.  The linearity assumption can best be tested with scatter plots.


**2. Multivariate normality**


The second assumption is of multivariate normality.  **Multivariate normality means that all the residuals are normally distributed. 
So, the errors between observed and predicted values (i.e., the residuals of the regression) should be normally distributed**. This assumption can best be checked with a histogram or a Q-Q-Plot.  Normality can be checked with a goodness of fit test, e.g., the Kolmogorov-Smirnov test.  When the data is not normally distributed a non-linear transformation (e.g., log-transformation) might fix this issue.


**3. No or little multi-collinearity**


Third assumption is that there is little or no multicollinearity in the data. Multicollinearity occurs when the independent variables are too highly correlated with each other. **So, it means that the independent variables are not correlated with each other.**
Multicollinearity may be checked in multiple ways as follows:-


1)	Correlation matrix – when computing the matrix of Pearson’s Bivariate Correlation among all independent variables the correlation coefficients need to be smaller than 1.


2)	Variance Inflation Factor (VIF) – the variance inflation factor of the linear regression is  
defined as VIF = 1/T. With VIF > 10 there is an indication that multicollinearity may be present; with VIF > 100 there is certainly multicollinearity among the variables.


If multicollinearity is found in the data, centering the data (that is deducting the mean of the variable from each score) might help to solve the problem.  However, the simplest way to address the problem is to remove independent variables with high VIF values.



**4. No auto-correlation in residuals**


Fourth assumption requires that there is no autocorrelation in the data.  Autocorrelation occurs when the residuals are dependent on each other. **So, it means that the error terms or residuals are independent from each other.**


While a scatterplot allows you to check for autocorrelations, you can test the linear regression model for autocorrelation with the Durbin-Watson test.  Durbin-Watson’s d tests the null hypothesis that the residuals are not linearly auto-correlated.  


**5. Homoscedasticity**


The last assumption of the linear regression analysis is homoscedasticity. It means that the residuals have same or constant variance. So the residuals are equal across the regression line. A scatterplot of residuals versus predicted values is good way to check for homoscedasticity.  There should be no clear pattern in the distribution.


=================================================================================


## 5. Dataset description


Now, we should get to know more about the dataset. It is a computer hardware dataset. The dataset consists of information about the computer vendors selling computers, model name of computers and various attributes to estimate the relative performance of CPU.


The dataset can be found at the following url –


https://archive.ics.uci.edu/ml/datasets/Computer+Hardware


The dataset description will help us to know more about the data.


**Dataset description** is given as follows:-


1. vendor name: 30 

      (adviser, amdahl,apollo, basf, bti, burroughs, c.r.d, cambex, cdc, dec, 
       dg, formation, four-phase, gould, honeywell, hp, ibm, ipl, magnuson, 
       microdata, nas, ncr, nixdorf, perkin-elmer, prime, siemens, sperry, 
       sratus, wang)
       
2. Model Name: many unique symbols

3. MYCT: machine cycle time in nanoseconds (integer)

4. MMIN: minimum main memory in kilobytes (integer)

5. MMAX: maximum main memory in kilobytes (integer)

6. CACH: cache memory in kilobytes (integer)

7. CHMIN: minimum channels in units (integer)

8. CHMAX: maximum channels in units (integer)

9. PRP: published relative performance (integer)

10. ERP: estimated relative performance from the original article (integer)
  

=================================================================================


## 6. The problem statement

A machine learning model is built with the aim of solving a problem. So, first of all I have to define the problem to be solved in this project.


As described earlier, the problem is to estimate the relative CPU performance of computer hardware dataset. Relative CPU performance of the computer hardware is described in terms of machine cycle time, main memory, cache memory and minimum and maximum channels as given in the dataset.


=================================================================================


## 7. Import the Python libraries


**to handle datasets**

`import numpy as np`

`import pandas as pd`


**for plotting**

`import matplotlib.pyplot as plt`

`% matplotlib inline`

`import seaborn as sns`


**import warnings**

`warnings.filterwarnings('ignore')`


=================================================================================


## 8. Import the dataset


`filename = "c:/datasets/machine.data.csv"`


`df = pd.read_csv(filename, header = None)`


================================================================================


## 9. Exploratory Data Analysis


Now, I will perform Exploratory Data Analysis. It provides useful insights into the dataset which is important for further analysis.
First of all, we should check the dimensions of the dataframe as follows:-


**view the dimensions of dataframe df**


`print("Shape of dataframe df: {}".format(df.shape))`


**view the top five rows of dataframe df with df.head() method**


`df.head()`


**rename columns of dataframe df**


`col_names = ['Vendor Name','Model Name', 'MYCT', 'MMIN', 'MMAX', 'CACH','CHMIN', 'CHMAX', 'PRP', 'ERP' ]`


`df.columns = col_names`


**view the top five rows of dataframe with column names renamed**


`df.head()`



### Explore types of variables

In this section, I will explore the types of variables in the dataset.

First, let's view a concise summary of the dataframe with `df.info()` method.


**view dataframe summary**

`df.info()`


There are categorical and numerical variables in the dataset. Numerical variables have data types int64 and categorical variables 
are those of type object.


First, let's explore the categorical variables.


**find categorical variables**


`categorical = [col for col in df.columns if df[col].dtype=='O']`


`print('There are {} categorical variables'.format(len(categorical)))`


**view the categorical variables**

`print(categorical)`


There are two categorical variables - **Vendor Name** and **Model Name** in the dataset.


The **Model Name** is a unique identifier for each of the computer models. Thus this is not a variable that we can use to predict the estimated relative performance of computer models. So, we should not use this column for model building.


Now, let's explore the numerical variables.


** find numerical variables**

`numerical = [col for col in df.columns if df[col].dtype!='O']`

`print('There are {} numerical variables'.format(len(numerical)))`


There are eight numerical variables in the dataset. They are 

['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']




**Summary : types of variables**


- There are 2 categorical variables and 8 numerical variables.

- The 2 categorical variables, **Vendor Name** and **Model Name** are 2 non-predictive attributes as given in the dataset description. So, I do not use them for model building.

- All of the 8 numerical variables are of discrete type.

- Out of the 8 numerical variables, **PRP** is the linear regression's guess. It is redundant column. I do not use it for model building.

- **ERP** (estimated relative performance is the goal field). It is the target variable.





**Estimate correlation coefficients**


Our dataset is very small. So, we can compute the standard correlation coefficient (also called Pearson's r) between every pair of attributes. 

We can compute it using the `df.corr()` method as follows:-



**estimate correlation coefficients**


`pd.options.display.float_format = '{:,.4f}'.format`


`corr_matrix = df.corr()`


`corr_matrix`



**Interpretation of correlation coefficient**


The correlation coefficient ranges from -1 to +1. 


When it is close to +1, this signifies that there is a strong positive correlation. So, we can see that there is a strong positive correlation between `ERP` and `MMAX`. 


When it is clsoe to -1, it means that there is a strong negative correlation. So, there is a small negative correlation between `ERP` and `MYCT`.




**Correlation heat map**


`plt.figure(figsize=(16,10))`


`plt.title('Correlation of Attributes with ERP')`


`a = sns.heatmap(corr_matrix, square=True, annot=True, fmt='.2f', linecolor='white')`


`a.set_xticklabels(a.get_xticklabels(), rotation=90)`


`a.set_yticklabels(a.get_yticklabels(), rotation=30)`


`plt.show()`


==================================================================================


## 10. Detect problems within variables


**Detect missing values**


let's visualise the number of missing values


`df.isnull().sum()`


We can confirm that there are no missing values in the dataset.


**Outliers in discrete variables**


let's view the summary statistics of the dataset


`df.describe()`


**detect outliers in discrete variables**


`for var in ['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX']:`

    `plt.figure(figsize=(16,10))`
    
    `(df.groupby(var)[var].count() / np.float(len(df))).plot.bar()`
    
    `plt.ylabel('Percentage of observations per label')`
    
    `plt.title(var)`
    
    `plt.show()`
    
    
    
The discrete variables show values that are shared by a tiny proportion of variable values in the dataset. For linear regression modeling, this does not cause any problem.



**Number of labels: cardinality**


Now, I will examine the categorical variable **Vendor Name**. First I will determine whether it show high cardinality. This is a high number of labels.


**plot the categorical variable**


`plt.figure(figsize=(12,8))`

`(df['Vendor Name'].value_counts()).plot.bar()`

`plt.title('Number of categories in Vendor Name variable')`

`plt.xlabel('Vendor Name')`

`plt.ylabel('Number of different categories')`

`plt.show()`


We can see that the **Vendor Name** variable, contain only a few labels. So, we do not have to deal with high cardinality.


=================================================================================


## 11. Linear Regression modeling


Now, I discuss the most important part of this project which is the Linear Regression model building. 


First of all, I will divide the dataset into categorical and numerical variables as follows:-


**Divide the dataset into categorical and numerical variables**


`df_cat = df.iloc[:,:2]`


`df_num = df.iloc[:, 2:]`


**Select the predictor and target variables**


`X = df_num.iloc[:,0:6]`


`y = df_num.iloc[:,-1]`


**Create separate train and test sets**


`from sklearn.model_selection import train_test_split`


`X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)`


**Feature Scaling**


Feature Scaling - I use the StandardScaler from sklearn


import the StandardScaler class from preprocessing library


`from sklearn.preprocessing import StandardScaler`


instantiate an object scaler


`scaler = StandardScaler()`


fit the scaler to the training set and then transform it


`X_train = scaler.fit_transform(X_train)`


transform the test set


`X_test = scaler.transform(X_test)`


The scaler is now ready, we can use it in a machine learning algorithm when required.



**Fit the Linear Regression model**


import the LinearRegression class from linear_model library


`from sklearn.linear_model import LinearRegression`


instantiate an object lr


`lr = LinearRegression()`


train the model using the training sets


`lr.fit(X_train, y_train)`


I instantiate the regressor lr and fit it on the training set with the fit method. In this step, the model learned the correlations between the training set (X_train, y_train). 


=================================================================================


## 12. Predicting the results


I have built the linear regression model. Now it is time to predict the results. I predict on the test set using the predict method. 


**Predicting the test set results**

We can predict the test set result as follows:-


`y_pred = lr.predict(X_test)`


**Predicting estimated relative CPU performance values**


`print("Predicted ERP - estimated relative performance for the first five values")`


`lr.predict(X_test)[0:5]`


=================================================================================


## 13. Model slope and intercept terms


The slope parameters(w) are also called weights or coefficients. They are stored in the **coef_** attribute.


The offset or intercept(b) is stored in the **intercept_** attribute.


So, the model slope is given by **lr.coef_** and model intercept term is given by **lr.intercept_**.


=================================================================================


## 14. Evaluate model performance

I have built the linear regression model and use it to predict the results. Now, it is the time to evaluate the model performance. 
We want to understand the outcome of our model and we want to know whether the performance is acceptable or not. 


For regression problems, there are several ways to evaluate the model performance. These are listed below:-


	- RMSE (Root Mean Square Error)
	
	- R2 Score
	
	- Check for Overfitting or Underfitting
	
	- Cross validation
	
	- Residual analysis
	
	- Normality test


I have described these measures in following sections:-


### i. RMSE


RMSE stands for **Root Mean Square Error**. RMSE is the standard deviation of the residuals. So, RMSE gives us the standard deviation 
of the unexplained variance by the model. It can be calculated by taking square root of Mean Squared Error.


RMSE is an absolute measure of fit. It gives us how spread the residuals are, given by the standard deviation of the residuals. 
The more concentrated the data is around the regression line, the lower the residuals and hence lower the standard deviation of residuals. It results in lower values of RMSE. So, lower values of RMSE indicate better fit of data. 


RMSE value is found to be 37.99.


### Interpretation

The RMSE value has been found to be 37.99. It means the standard deviation for our prediction is 37.99. So, sometimes we expect the predictions to be off by more than 37.99 and other times we expect less than 37.99.


### ii. R2 Score


**R2 Score** is another metric to evaluate performance of a regression model. It is also called **Coefficient of Determination**. 
It gives us an idea of goodness of fit for the linear regression models. It indicates the percentage of variance that is explained by the model. 


**R2 Score = Explained Variation/Total Variation**


Mathematically, we have


$$R^2=1-\frac{SS_{res}}{SS_{tot}}$$


 The total sum of squares, $SS_{tot}=\sum_i(y_i-\bar{y})^2$


 The regression sum of squares (explained sum of squares), $SS_{reg}=\sum_i(f_i-\bar{y})^2$


 The sum of squares of residuals (residual sum of squares), $SS_{res}=\sum_i(y_i-f_i)^2 = \sum_ie^2_i$


In general, the higher the R2 Score value, the better the model fits the data. Usually, its value ranges from 0 to 1. So, we want its value to be as close to 1. Its value can become negative if our model is wrong.


R2 Score value is found to be 0.92.


### Interpretation


In business decisions, the benchmark for the R2 score value is 0.7. It means if R2 score value >= 0.7, then the model is good enough 
to deploy on unseen data whereas if R2 score value < 0.7, then the model is not good enough to deploy. 


Our R2 score value has been found to be 0.92. It means that this model explains 92% of the variance in our dependent variable. So, 
the R2 score value confirms that the model is good enough to deploy because it provides good fit to the data.


### iii. Overfitting Vs Underfitting


Evaluating training set performance


`print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))`


Training set score: 0.91


Evaluating test set performance


`print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))`


Test set score: 0.92



### Interpretation 

Training and test set performances are comparable. An R Square value of 0.92 is very good.



### iv. Cross validation


Cross-validation is a vital step in evaluating a model. It maximizes the amount of data that is used to train the model. 

In cross-validation, we split the training data into several subgroups. Then we use each of them in turn to evaluate the model 
fitted on the remaining portion of the data.

It helps us to obtain reliable estimates of the model's generalization performance. So, it helps us to understand how well the model performs on unseen data.

The average 5-Fold cross validation score is found to be 0.4691


**Interpretation**


There is a large fluctuation in the cross validation scores of the model. 

The average 5-fold cross validation score is very poor and hence the linear regression model is not a great fit to the data.



### v. Residual analysis


A linear regression model may not represent the data appropriately. The model may be a poor fit to the data. So, we should validate 
our model by defining and examining residual plots. The difference between the observed value of the dependent variable (y) and the predicted value (ŷi) is called the **residual** and is denoted by **e**. The scatter-plot of these residuals is called **residual plot**.


If the data points in a residual plot are randomly dispersed around horizontal axis and an approximate zero residual mean, a linear regression model may be appropriate for the data. Otherwise a non-linear model may be more appropriate.


### Interpretation 


A regression model that has nicely fit the data will have its residuals display randomness (i.e., lack of any pattern). This comes 
from the **homoscedasticity** assumption of regression modeling. Typically scatter plots between residuals and predictors are used to confirm the assumption. Any pattern in the scatter-plot, results in a violation of this property and points towards a poor fitting model.


Residual errors plot show that the data is randomly scattered around line zero. The plot does not display any pattern in the residuals.  Hence, we can conclude that the Linear Regression model is a good fit to the data.


### vi. Normality test (Q-Q Plot)


This is a visual or graphical test to check for normality of the data. This test helps us identify outliers and skewness. The test is performed by plotting the data verses theoretical quartiles. The same data is also plotted on a histogram to confirm normality.


Any deviation from the straight line in normal plot or skewness/multi-modality in histogram shows that the data does not pass the normality test.


**Interpretation**

From the distribution plots, we can see that all the above variables are positively skewed. The Q-Q plot of all the variables confirm that the variables are not normally distributed.

Hence, the variables do not pass the normality test.


=================================================================================


## 15. Conclusion


I carry out residual analysis to check for homoscedasticity assumption. Residual errors plot show that the data is randomly scattered around line zero. The plot does not display any pattern in the residuals.  Hence, we can conclude that the Linear Regression model is a good fit to the data.


The R-squared or the coefficient of determination is 0.4691 on an average for 5-fold cross validation. It means that the predictor is only able to explain 46.91% of the variance in the target variable. This indicates that the model is not a good fit to the data.


I carry out normality test to check for distribution of the variables. We can see that the variables do not follow the normal distribution. The Q-Q plots confirm the same.


So, we can conclude that the linear regression model is unable to model the data to generate decent results. It should be noted that 
the model is performing equally on both training and testing datasets. It seems like a case where we would need to model this data 
using methods that can model non-linear relationships. Also variables need to be transformed to satisfy the normality assumption.


=================================================================================


## 16. References


The concepts and ideas in this project have been taken from the following websites and books:-


i.	Python Data Science Handbook by Jake VanderPlas


ii.	Hands-On Machine Learning with Scikit Learn and Tensorflow by Aurlien Geron


iii.	Introduction to Machine Learning with Python by Andreas C Muller and Sarah Guido


iv.	https://en.wikipedia.org/wiki/Linear_regression


v.	https://en.wikipedia.org/wiki/Simple_linear_regression


vi.	https://en.wikipedia.org/wiki/Ordinary_least_squares


vii.	https://en.wikipedia.org/wiki/Root-mean-square_deviation


viii.	https://en.wikipedia.org/wiki/Coefficient_of_determination


ix.	https://www.statisticssolutions.com/assumptions-of-linear-regression/


