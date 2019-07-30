---
layout:     post
title:      "A Brief Introduction to Supervised Learning"
date:       2019-03-28 12:00:00
author:     "A.I. Dan"
---

# A Brief Introduction to Supervised Learning

<b>Summary:</b> In this post I will discuss the details of <b>supervised</b> machine learning and its applications. Code examples will be used for demonstration but the theory and background knowledge will be the main focus.

Supervised learning is the most common subbranch of machine learning today. Typically, new machine learning practitioners will begin their journey with supervised learning algorithms. Therefore, the first of this three post series will be about supervised learning.

<hr>

Supervised machine learning algorithms are designed to learn by example. The name "supervised" learning originates from the idea that training this type of algorithm is like having a teacher supervise the whole process.

When training a supervised learning algorithm, the training data will consist of inputs paired with the correct outputs. During training, the algorithm will search for patterns in the data that correlate with the desired outputs. After training, a supervised learning algorithm will take in new unseen inputs and will determine which label the new inputs will be classified as based on prior training data. The objective of a supervised learning model is to predict the correct label for newly presented input data. At its most basic form, a supervised learning algorithm can be written simply as:

<img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;Y&space;=&space;f(x)" title="Y = f(x)" style='display: block; margin: auto;'>

Where *Y* is the predicted output that is determined by a mapping function that assigns a class to an input value *x*. The function used to connect input features to a predicted output is created by the machine learning model during training.

Supervised learning can be split into two subcategories: <b>Classification</b> and <b>regression</b>.

### Classification

<img src='a-i-dan.github.io/images/supervised_learning_post/supervised learning post.png?raw=true' style='margin: auto; display: block;'>

During training, a classification algorithm will be given data points with an assigned category. The job of a classification algorithm is to then take an input value and assign it a class, or category, that it fits into based on the training data provided.

The most common example of classification is determining if an email is spam or not. With two classes to choose from (spam, or not spam), this problem is called a binary classification problem. The algorithm will be given training data with emails that are both spam and not spam. The model will find the features within the data that correlate to either class and create the mapping function mentioned earlier: *Y=f(x)*. Then, when provided with an unseen email, the model will use this function to determine whether or not the email is spam.

Classification problems can be solved with a numerous amount of algorithms. Which ever algorithm you choose to use depends on the data and the situation. Here are a few popular classification algorithms:
  * Linear Classifiers
  * Support Vector Machines
  * Decision Trees
  * K-Nearest Neighbor
  * Random Forest

### Regression

Regression is a predictive statistical process where the model attempts to find the important relationship between dependent and independent variables. The goal of a regression algorithm is to predict a continuous number such as sales, income, and test scores. The equation for basic linear regression can be written as so:


<img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\hat&space;y&space;=&space;w[0]&space;*&space;x[0]&space;&plus;&space;w[1]&space;*&space;x[1]&space;&plus;&space;\cdots&space;&plus;&space;w[i]&space;*&space;x[i]&space;&plus;&space;b" title="\hat y = w[0] * x[0] + w[1] * x[1] + \cdots + w[i] * x[i] + b" style='display: block; margin: auto;'>

Where *x[i]* is the feature(s) for the data and where *w[i]* and *b* are parameters which are developed during training. For simple linear regression models with only one feature in the data, the formula looks like this:

<img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\hat&space;y&space;=&space;wx&space;&plus;&space;b" title="\hat y = wx + b" style='display: block; margin: auto;'/>

Where *w* is the slope, *x* is the single feature and *b* is the y-intercept. Familiar? For simple regression problems such as this, the models predictions are represented by the line of best fit. For models using two features, the plane will be used. Finally, for a model using more than two features, a hyperplane will be used.

Imagine we want to determine a student's test grade based on how many hours they studied the week of the test. Lets say the plotted data with a line of best fit looks like this:

<img src='a-i-dan.github.io/images/supervised_learning_post/supervised learning post2.png?raw=true' style='margin: auto; display: block;'>

There is a clear positive correlation between hours studied (independent variable) and the student's final test score (dependent variable). A line of best fit can be drawn through the data points to show the models predictions when given a new input. Say we wanted to know how well a student would do with five hours of studying. We can use the line of best fit to predict the test score based on other student's performances.

There are many different types of regression algorithms. The three most common are listed below:
  * Linear Regression
  * Logistic Regression
  * Polynomial Regression

### Simple Regression Example

First we will import the needed libraries and then create a random dataset with an increasing output.

```python
# Regression Example
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

X, y = make_regression(n_samples=100, n_features=1, noise=13)

plt.scatter(X,y)
plt.show()
```
Output:
<img src='a-i-dan.github.io/images/supervised_learning_post/regression_ex_data_scatter.png?raw=true' style='margin: auto; display: block;'>

We can then place our line of best fit onto the plot along with all of the data points.

```python
reg = LinearRegression(fit_intercept=True)
reg.fit(X, y)

xfit = np.linspace(start=-2, stop =3)
yfit = model.predict(xfit[:, np.newaxis])

plt.scatter(X, y)
plt.plot(xfit, yfit)
```
Output:

<img src='a-i-dan.github.io/images/supervised_learning_post/reg_data_with_line.png?raw=true' style='margin: auto; display: block;'>

We will then print out the slope and intercept of the regression model.

```python
print("Slope:    ", reg.coef_[0])
print("Intercept:", reg.intercept_)
```
Output:
```
Slope:     65.54726684409927
Intercept: -1.8464500230055103
```

In middle school, we all learned that the equation for a linear line is *y = mx + b*. We can now create a function called "predict" that will multiply the slope (*w*) with the new input (*x*). This function will also use the intercept (*b*) to return an output value. After creating the function, we can predict the output values when *x = 3* and when *x = -1.5*.

```python
# y = wx + b  -or-  y = mx + b
def predict(x):
    w = reg.coef_[0]
    b = reg.intercept_
    y = w*x + b
    return y

print("Predict y For 3:     ", predict(3))
print("Predict y For -1.5: ", predict(-1.5))
```
Output:
```
Predict y For 3:      194.7953505092923
Predict y For -1.5:  -100.16735028915441
```
Now let's plot the original data points with the line of best fit. We can then add the new points that we predicted (colored red). As expected, they fall on the line of best fit.

```python
plt.scatter(X, y)
plt.plot(xfit, yfit)
plt.plot(3, predict(3), 'ro')
plt.plot(-1.5, predict(-1.5), 'ro')
```
Output:
<img src='a-i-dan.github.io/images/supervised_learning_post/reg_red.png?raw=true' style='margin: auto; display: block;'>

### Conclusion

Supervised learning is the simplest subcategory of machine learning and serves as an introduction to machine learning to many machine learning practitioners. Supervised learning is the most commonly used form of machine learning, and has proven to be an excellent tool in many fields. This post was part one of a three part series. Part two will cover **unsupervised learning**.
