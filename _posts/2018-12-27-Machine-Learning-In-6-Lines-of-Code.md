# Machine Learning in 6 Lines of Code


In this post I will be reviewing a simple machine learning (ML) model. Our data set for this model will be three species of iris flowers with their pedal and sepal measurements.

<hr>

<h3>Supervised Learning</h3>

This is an example of <b>supervised learning</b> because we have labeled data to measure the success of our models predictions. In other words, we have a dataset that has both our input measurements and also our desired output. We can train our model on the correctly labeled data, then test the model with only inputs. This way, when we want to measure our models accuracy, we can go back and see if our test data matches the predictions.

I plan to write about the different types of ML in a future post.

The goal of creating this ML model is to distinguish between the three species of iris flowers. The three species are <b>setosa</b>, <b>versicolor</b>, and <b>virginica</b>. We want our model to learn from the measurements of the flowers. We then want to give our model a set of new inputs (new measurements) and have it predict the correct output (species name).

<h5>Let's get started...</h5>

Here is the code we will be working with:

<!--<script src="https://gist.github.com/A-I-dan/c1852b9950c00850a4e59fa675646b9d.js"></script>-->
```python
iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
y_prediction = knn.predict(X_test)
print('Accuracy: {:2f}'.format(np.mean(y_prediction == y_test)))
```


<b>Note</b>: For this model we will be using scikit-learn. Scikit-learn, in my opinion, would not be my first choice for learning what goes on underneath the hood of ML models. While I usually prefer to learn the "under the hood" part of the model, for this example we will not be going over that for simplicity reasons.

<hr>

<h5>Now let's really get started...</h5>

I will start off by first visualizing the dataset we will be working with.

<b>Plots</b>:

<img src='https://github.com/A-I-dan/blog/blob/master/images/iris_dataset_plot.png?raw=true'>


The scatter matrix will show three colors, each representing a species:

<b>Blue</b>: Setosa, <b>Green</b>: Virginica, <b>Orange</b>: Versicolor.

Along both the x-axis and y-axis there will be four labels for our measurements. Sepal length, sepal width, petal width and petal length.

If you look at the plots, you will notice that some have more distinct grouping than others. Look at the <b>setosa</b> species. In all of the graphs it looks like the <b>setosa</b> species (blue) is completely separated from the other two species. Some plots have points that mix in together while some plots will show a clear difference between species.

<hr>

<h5>Now the code...</h5>

I will start off by explaining the needed Python packages to make this model work.

Here are the packages you will need:

<!--<script src="https://gist.github.com/A-I-dan/43dc749f03a5af805d88817dd774a3fa.js"></script>-->
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
```

<b>(Line 1)</b>: `from sklearn.datasets import load_iris`

The data for our model is coming from the first line in our code. We are pulling from the datasets and taking out `load_iris`. Later, I will show you how we bring the data into our model.

<b>(Line 2)</b>: `from sklearn.model_selection import train_test_split`

In the second line of our code, we have our split function that allows us to split our data into two groups. One group will consist of 75% of the data and will be our training set. Like the name suggests, we will be using the training set to train our model. The second group will consist of the remaining 25% of the data that we will later use to test our model.

<b>(Line 3)</b>: `from sklearn.neighbors import KNeighborsClassifier`

This is where the magic starts happening. For our model, our choice of a classification algorithm will be a <b>k-nearest neighbors classifier</b> (one of the simplest ML algorithms). This algorithm will help us make our predictions by finding the closest related data point to the unknown inputs we give it. For example, if we give it measurements to a new flower, it will find the most similar point within the training data and give it that label. It finds the "nearest neighbor" in the data.

<b>(Line 4)</b>: `import numpy as np`

NumPy is one of the most important Python packages to have and is one of the most commonly used. NumPy is a linear algebra library that is useful for its n-dimensional arrays for data storage.

<hr>

<h5>Now we start the fun part...</h5>


These next six lines of code are where we will create and test our model.

Here is the code:

<!--<script src="https://gist.github.com/A-I-dan/79b8f87802bc13ad418447f4d6214112.js"></script>-->
```python
iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state = 0)

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

y_prediction = knn.predict(X_test)

print('Accuracy: {:2f}'.format(np.mean(y_prediction == y_test)))
```

<b>(Line 1)</b> `iris_dataset = load_iris()`

As mentioned earlier when explaining why we import `load_iris` in this line of our code: `from sklearn.datasets import load_iris`. This is the data for our model. Here, we are taking `load_iris` and setting it to `iris_dataset`. This will contain our data's values.


<b>(Line 2)</b>  `X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state = 0)`

  This is where we utilize the `train_test_split` function that I mentioned earlier as well. The `train_test_split` function will randomly select data points and split them among the two groups (One with 75% of the data and one with 25%).

  `X_train`, `X_test`, `y_train` and `y_test` will all be NumPy arrays.

  `X_train`: contains 75% of the measurements in our data.

  `X_test`: contains 25% of the measurements in our data.

  `y_train`: contains 75% of the outputs corresponding `X_train`.

  `y_test`: contains 25% of the outputs corresponding to `x_test`.

  The `random_state = 0` will make sure that this line always has the same outcome everytime we run the code. If the random_state function is not specified, then every time we run that line of code we will generate a different random set of data for our two groups (training set and testing set).

<b>(Line 3)</b> `knn = KNeighborsClassifier(n_neighbors = 1)`

This is where we bring in the <b>k-nearest neighbors classifier</b> algorithm that we imported earlier. `n_neighbors = 1` is saying that the algorithm will use the first closest neighbor to the new data point. The algorithm will only search for, and base the prediction off of, one neighbor with n_neighbors set to one.

I would recommend playing around with setting `n_neighbors` to different numbers and watching how it affects the accuracy.

We are setting the <b>k-nearest neighbors classifier</b> algorithm to `knn` to hold that information.

<b>(Line 4)</b> `knn.fit(X_train, y_train)`

This line is where we will be training our model. The `.fit()` function lets us fit our model to our data, which is essentially training our model.

In this line we are using our k-nearest neighbor algorithm, which we set to `knn`, and fitting our model to our `X_train` and `y_train` data. Once our model is trained on the training data, we can later call the `.predict()` function, which lets us make our predictions based on the training data.

<b>(Line 5)</b> `y_prediction = knn.predict(X_test)`

This is where the `.predict()` function comes into play. In this line of our code our model is making predictions for new data that we introduce to it. We are using our k-nearest neighbor algorithm to make predictions based on the X_test data (The X_test data was the 25% that we saved to later test out our model).

<b>Note</b>: Since we are feeding our model the X_test data, it will spit out a `y_prediction` based on the X_test data. Remember that our "X" data consists of the four measurements (sepal length, sepal width, petal length and petal width). And remember that our "y" data consists of the three names of the species (<b>setosa</b>, <b>versicolor</b>, and <b>virginica</b>).

If we give our model the measurements, we will get back the name of the species it is predicted to be.

That wraps up the actual building of the ML model, but lets check how accurate the predictions were.

<b>(Line 6)</b>: `print('Accuracy: {:2f}'.format(np.mean(y_prediction == y_test)))`

This final line of our code will print out the accuracy of our model.

For information on the `print()` function, see my "Hello, world!" blog post.

The `.format()` function will return a formatted representation of a given value. The brackets will create an empty dictionary where `:2f` inside the brackets means we will be rounding to the first two digits. For example, instead of getting an accuracy of 97.3684....% we will shorten it down to just 97%.

<hr>

Now that we have built our model for predicting the species of iris flowers, let's see it in action.

Here is the code from above with a few modified lines for this example:

<!--<script src="https://gist.github.com/A-I-dan/c0bdbb5c1db77bcc5b811c9dd5bae346.js"></script>-->
```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state = 0)

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

X_new_input = np.array([[4, 2.1, 2.2, 0.2]]) 
X_newer_input = np.array([[6.2, 3.4, 5.4, 2.3]]) 

prediction = knn.predict(X_new_input)
print('Predicted Target Name: {}'.format(iris_dataset['target_names'][prediction]))

prediction_2 = knn.predict(X_newer_input)
print('Predicted Target Name: {}'.format(iris_dataset['target_names'][prediction_2]))
```

The top 12 lines are the same as before, with the extras added in starting at <b>line 14</b>.

<b>(Line 14)</b>: `X_new_input = np.array([[4, 2.1, 2.2, 0.2]])`

Now that we know our model is working, and is 97% accurate, we can begin to make predictions on new measurements that we do not know the correct answer to. In this line, we are giving our model these four measurements:

Sepal Length: 4 cm

Sepal Width : 2.1 cm

Petal Length: 2.2 cm

Petal Width : 0.2 cm

We want our model to tell us what species these measurements would most likely be.

<b>(Line 15)</b>: `X_newer_input = np.array([[6.2, 3.4, 5.4, 2.3]])`

Line 15 is our second flower that we want to predict. This flower has different measurements than the first flower in <b>line 14</b>.

Sepal Length: 6.2 cm

Sepal Width : 3.4 cm

Petal Length: 5.4 cm

Petal Width : 2.3 cm

<b>(Line 17 - 18)</b>:
```
prediction = knn.predict(X_new_input)
print('Predicted Target Name:{}'.format(iris_dataset['target_names'][prediction]))
```
and

<b>(Line 20 - 21)</b>
```
prediction_2 = knn.predict(X_newer_input)
print('Predicted Target Name: {}'.format(iris_dataset['target_names'][prediction_2]))
```

I will be doing all four of these lines together because both sets are similar and can be explained together.

On <b>line 17</b>, we are using the `.predict()` function again (see <b>line 5</b>). Instead of calling the predict function on an entire set of data, we will use it for just one prediction. We want to predict what species would have the measurements from `X_new_input` (<b>line 14</b>). We are calling this prediction...`prediction`. Clever, I know.

The same goes for <b>line 15</b>, except there are different measurements involved.

On <b>line 18</b> and <b>line 21</b> we have our `print()` function again. We want to print the "Predicted Target Names" of the measurements we feed the model.

Again, the `.format()` function will return a formatted representation of a given value. We want our predictions to be in the form of the `target_names` from the iris_dataset. The target names are <b>setosa</b>, <b>virginica</b>, and <b>versicolor</b>.

The two brackets (`{}`) create an empty dictionary (this is where our target name will be placed).

The output to these four lines looks like this:
```
Predicted Target Name: ['setosa']
Predicted Target Name: ['virginica']
```
The measurements `[4, 2.1, 2.2, 0.2]` will give us <b>setosa</b>.

And the measurements `[6.2, 3.4, 5.4, 2.3]` will give us <b>virginica</b>.

<b>Remember</b>: These predictions are only 97% accurate.

<hr>

Review of <i>Introduction to Machine Learning with Python</i>.
