---
authors: balazskvancz
---

# 01 - Python AI Environments and Simple Neural Networks

## Goal

The main goal of this laboratory is to become familiar with the basic tools used for developing different artificial intelligence approaches.

The structure of this laboratory is as follows:

1. Installing the required dependencies.
1. Getting familiar with the used technologies through various examples.
1. Exercise One [3 points in total for completing all sub-exercises]
    1. NumPy exercise
    1. Pandas exercise
    1. Matplotlib exercise
1. Simple Linear Regression using PyTorch [4 points in total]
    1. Creating and visualizing the dataset [1 point]
    1. Building and training the model [2 points]
    1. Visualizing the training and validation loss [1 point]
1. Multiclass Softmax Classifier [6 points in total]
    1. Loading the dataset [1 point]
    1. Implementing the model [1 point]
    1. Building and training the model [3 points]
    1. Visualizing the training loss [1 point]
1. Handwritten Number Classifier [7 points in total]
    1. Loading the dataset using data loaders [2 points]
    1. Implementing the model [1 point]
    1. Building and training the model [3 points]
    1. Visualizing the training loss [1 point]
1. Giving feedback [+1 point]

!!! info "Grading"
    In order to pass this laboratory, you must obtain at least 8 points out of 20.

!!! important "Screenshot requirement"
    At the end of **every** exercise (marked with a :camera: **Screenshot** note), you must take a screenshot of your working solution, including its output, and upload it together with your solution, using the exact filename given in that note (e.g. `f1_1.png`, `f2.png`). Solutions missing the required screenshots, or using the wrong filename, will not be accepted.

## Preparation

Don't forget to follow the assignment submission process described under [GitHub](../../information/github.md) while working on this laboratory.

!!! important "Reviewer"
    When creating the pull request for this laboratory, assign it to the `balazskvancz` GitHub user.

!!! tip "Jupyter notebook or plain Python files"
    You may solve the exercises either in a Jupyter notebook or in plain `.py` files, whichever you prefer. Regardless of the format you choose, make sure that **everything** (code, and the required screenshots) is committed and pushed to your solution branch.

## Setup

!!! important "Mandatory"
    Issue the following command to install the necessary libraries.

    ```bash
    pip install numpy pandas matplotlib torch torchvision tensorflow
    ```

!!! tip "Optional"
    Install `jupyter` if you are working on your own machine.

    ```bash
    pip install jupyter
    ```

## NumPy introduction

The following code snippet acts as an introduction to the `NumPy` library.

```python
import numpy as np

# Simple one-dimensional arrays (also known as vectors)
# can be created easily as follows:
one_dim_array = np.array([1, 2, 3, 4, 5])
print("=====================")
print("One-dimensional array:")
print(one_dim_array)
print(f"=====================\n")

# Matrices can also be created by providing their
# rows as array parameters.
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("=====================")
print("Simple 2x3 matrix:")
print(matrix)
print(f"=====================\n")

# Sometimes, zero-filled matrices are needed.
# They can be created by providing the required shape
# as a tuple parameter.
zeros = np.zeros((3, 4))
print("=====================")
print("Zero-valued 3x4 matrix:")
print(zeros)
print(f"=====================\n")

# Similarly, this function creates a NumPy array
# filled with ones. This is useful when a matrix
# with the same constant value in every cell is needed.
ones = np.ones((2, 3))
print("=====================")
print("One-valued 2x3 matrix:")
print(ones)
print(f"=====================\n")

# NumPy provides a utility function to generate
# N evenly distributed values between X and Y.
lin = np.linspace(0, 10, 5)
print("=====================")
print("5 evenly distributed values between 0 and 10:")
print(lin)
print(f"=====================\n")

# The next function generates random integer values
# from a specified interval [low, high),
# meaning the lower bound is inclusive and the upper bound is exclusive.
# The shape of the expected tensor can also be provided.
rand_int = np.random.randint(0, 100, size=(3, 3))
print("=====================")
print("Randomly filled tensor with shape (3, 3):")
print(rand_int)
print(f"=====================\n")

arr = np.array([10, 20, 30, 40, 50])

# Common methods work on NumPy arrays,
# such as accessing elements by index
# or slicing the original array.
print("=====================")
print("The first element is:", arr[0])
print("The last element is:", arr[-1])

print("Slicing the original array with [1:4]:")
print(arr[1:4])
print(f"=====================\n")

# Indexing also works for multi-dimensional
# NumPy arrays.
# It is possible to access a single element
# by providing its row and column indices.
#
# Rows and columns can also be accessed separately.
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print("=====================")
print("matrix[0, 1]:", matrix[0, 1])

print("The first column of each row ([:, 0]):")
print(matrix[:, 0])

print("The first row of the matrix ([0, :]):")
print(matrix[0, :])
print(f"=====================\n")

# Useful mathematical operations
# are also implemented, for example:
#   - adding a constant to each element,
#   - multiplying each element by a constant,
#   - raising each element to a given power.
x = np.array([1, 2, 3, 4])

print("=====================")
print("Adding 10 to each element:")
print(x + 10)

print("Multiplying each element by 2:")
print(x * 2)

print("Raising each element to the power of 2:")
print(x ** 2)
print(f"=====================\n")

# Further mathematical functions are built in, such as:
print("=====================")
print("Mean:", np.mean(x))
print("Standard deviation:", np.std(x))
print("Sum:", np.sum(x))
print("Minimum:", np.min(x))
print("Maximum:", np.max(x))
print(f"=====================\n")
```

```python
# In addition to basic mathematical functions,
# NumPy provides several higher-level
# linear algebra functions, which are
# essential in the field of machine learning.
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# The np.dot() function computes the matrix
# product of two compatible matrices.
C = np.dot(A, B)
print("=====================")
print("Dot product of A and B:")
print(C)
print(f"=====================\n")

# Transposition swaps the rows and columns of a matrix.
print("=====================")
print("Transpose of A:")
print(A.T)
print(f"=====================\n")

# If the given matrix is invertible,
# its inverse can be computed
# using NumPy's linear algebra module.
invA = np.linalg.inv(A)
print("=====================")
print("Inverse matrix of A:")
print(invA)
print(f"=====================\n")
```

## Pandas introduction

The following code snippet acts as an introduction to the `Pandas` library.

```python
import pandas as pd

data = {
    "Name": ["Anna", "Adam", "Cecile", "Dora"],
    "Age": [22, 25, 23, 24],
    "Score": [85, 90, 78, 92]
}

df = pd.DataFrame(data)
print(df)
```

```python
# Printing the first two rows of the DataFrame.
print("=====================")
print("Printing the first two rows:")
print(df.head(2))
print(f"=====================\n")

# Printing a single column.
print("=====================")
print("Printing only the Age column:")
print(df["Age"])
print(f"=====================\n")

# Printing multiple columns
# at the same time.
print("=====================")
print("Printing only the Name and Score columns:")
print(df[["Name", "Score"]])
print(f"=====================\n")

# Rows can be selected using indexing.
print("=====================")
print("The row at index [1]:")
print(df.iloc[1])
print(f"=====================\n")

# It is possible to query data
# in a declarative manner by filtering
# rows based on a given predicate.
print("=====================")
print("Rows with Score greater than 85:")
print(df[df["Score"] > 85])
print(f"=====================\n")

print("=====================")
print("The mean of the Score values:", df["Score"].mean())
print(f"=====================\n")

print("=====================")
print("The maximum of the Score values:", df["Score"].max())
print(f"=====================\n")

# Descriptive statistics include measures that summarize
# the central tendency, dispersion, and shape
# of a dataset's distribution.
print("=====================")
print(df.describe())
print(f"=====================\n")
```

## Matplotlib introduction

The following code snippet acts as an introduction to the `Matplotlib` library.

```python
import matplotlib.pyplot as plt

# The following snippet plots
# the function y = x^2 for
# 50 evenly spaced values between 0 and 10.
x = np.linspace(0, 10, 50)
y = x ** 2

plt.plot(x, y)
plt.title("y = x^2")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()
```

```python
# This snippet illustrates that multiple
# plots can be visualised with different labels.
y1 = x
y2 = x**2
y3 = x**3

plt.plot(x, y1, label="y = x")
plt.plot(x, y2, label="y = x^2")
plt.plot(x, y3, label="y = x^3")

plt.legend()
plt.title("Multiple function")
plt.show()
```

```python
# Apart from line diagrams,
# scatter plots can be created as well.
#
# The following snippet draws
# points based on random distribution.
x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x, y)
plt.title("Random points")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

```python
names = ["Anna", "Adam", "Cecile"]
scores = [85, 90, 78]

plt.bar(names, scores)
plt.title("Scores")
plt.ylabel("Score")
plt.show()
```

## Introduction to scikit-learn

In this laboratory, we also introduce the `scikit-learn` library, which has not been covered in detail during the lectures.

`scikit-learn` is one of the most widely used Python libraries for machine learning. It provides simple and efficient tools for:

- data preprocessing,
- dataset splitting,
- feature scaling,
- classification, regression, and clustering,
- and model evaluation.

The goal of this laboratory is to give students practical experience with a commonly used machine learning framework and to complement the theoretical knowledge gained during the lectures. Through hands-on exercises, students will learn how to:

- load and preprocess datasets,
- split data into training and test sets,
- apply standard scaling techniques,
- and evaluate machine learning models using built-in tools.

This introduction prepares students for working with real-world machine learning pipelines and helps bridge the gap between theory and practice.

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the built-in diabetes dataset from scikit-learn.
# This dataset contains medical features and a continuous target value.
data = load_diabetes()

X = data.data
y = data.target

# Create a Pandas DataFrame for easier data inspection and handling.
df = pd.DataFrame(X, columns=data.feature_names)
df["target"] = y

# Split the dataset into training and testing sets.
# 80% of the data is used for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples count:", X_train.shape[0])
print("Test samples count:", X_test.shape[0])

# Create a StandardScaler to normalize the feature values.
# This ensures that all features have mean 0 and standard deviation 1.
scaler = StandardScaler()

# Fit the scaler on the training data and transform it.
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same transformation to the test data.
X_test_scaled = scaler.transform(X_test)

# Create a Linear Regression model.
model = LinearRegression()

# Train the model using the scaled training data.
model.fit(X_train_scaled, y_train)

# Predict target values for the test set.
y_pred = model.predict(X_test_scaled)

# Evaluate the model using common regression metrics.
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² score:", r2)
```

## Exercise 1

### Exercise 1.1

- Create a function called `exercise_1_1` that takes an integer as input.
- Generate an array of the given size with elements sampled from a normal distribution.
- Calculate the following values:
    - mean,
    - standard deviation,
    - maximum value,
    - minimum value.
- Return these values as a dictionary: `{"mean", "std", "min", "max"}`.

```python
import numpy as np
from typing import Dict

def exercise_1_1(n: int) -> Dict[str, float]:
  # TODO
  pass
```

!!! note ":camera: Screenshot"
    Take a screenshot of the output of `exercise_1_1` and upload it together with your solution as `f1_1.png`.

### Exercise 1.2

- Create a function called `exercise_1_2` that takes no arguments.
- Generate 100 data points between 0 and 2π using `np.linspace`.
- Calculate the values of `y = sin(x)` using NumPy.
- Calculate the values of `y = cos(x)` using NumPy.
- Plot these values using Matplotlib.
- Give the plot an appropriate title.
- Display a corresponding legend.
- Visualize grid lines.

```python
import numpy as np
import matplotlib.pyplot as plt

def exercise_1_2():
  # TODO
  pass
```

!!! note ":camera: Screenshot"
    Take a screenshot of the resulting plot produced by `exercise_1_2` and upload it together with your solution as `f1_2.png`.

### Exercise 1.3

- Create a function called `exercise_1_3` that takes no arguments.
- Load the `vgsales.csv` file into a Pandas DataFrame.
- Convert the `Genre` column to the categorical (category) data type for optimization.
- Group the data by the `Genre` column.
- For each genre, calculate:
    - the average global sales (`Global_Sales`),
    - the total North America sales (`NA_Sales`).
- Keep only those genres where the average global sales is greater than `0.5`.
- Use index matching to:
    - select the favourite genres based on the `FAVOURITE_GENRES` list,
    - and select their complement set (non-favourite genres).
- Sort the non-favourite genres by total `NA_Sales` sales in descending order.
- Return:
    - the top 3 non-favourite genres by `NA_Sales`,
    - and the DataFrame containing the favourite genres.

#### Downloading the dataset

Run the following command. It downloads the necessary dataset and unzips it.

```bash
curl -L -o video_games.zip https://www.kaggle.com/api/v1/datasets/download/gregorut/videogamesales && unzip video_games.zip && rm -rf video_games.zip
```

```python
import pandas as pd

VG_SALES_FILE_PATH="vgsales.csv"
FAVOURITE_GENRES = ["Action", "Sports", "Racing"]

def exercise_1_3():
  # TODO
  pass
```

!!! note ":camera: Screenshot"
    Take a screenshot of the output of `exercise_1_3` and upload it together with your solution as `f1_3.png`.

## Exercise 2 – Simple Linear Regression

In this exercise, the task is to train and test a simple linear regression model. The used dataset will be synthetic before moving on to real-life problems.

The goal is to find a slope that fits the following relationship:

y = 2x + 1 + ϵ

- Create a vector `x` of 1000 evenly spaced values between 0 and 100. Make sure that `ϵ` is sampled from a normal distribution with a standard deviation of 2.
- Visualize the dataset as a scatter plot.
- Split the generated dataset into training, validation, and testing sets using the following proportions: 60%, 20%, 20%.
- Train a model that fits the data.
- Visualise the training and validation loss during the process.

```python
import numpy as np

min = 0
max = 100
std_dev = 2
no_elements = 100

def create_dummy_dataset():
  # TODO
  pass
```

```python
import matplotlib.pyplot as plt

def visualise_dummy_dataset(X, y):
  # TODO
  pass
```

```python
X, y = create_dummy_dataset()
```

```python
visualise_dummy_dataset(X, y)
```

!!! note ":camera: Screenshot"
    Take a screenshot of the visualized dataset and upload it together with your solution as `f2_1.png`.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def fit_simple_linear_regression(epochs = 10):
  X, y = create_dummy_dataset()
  # TODO
  pass
```

```python
# TODO: run this after implementing the function.
fit_simple_linear_regression()
```

!!! note ":camera: Screenshot"
    Take a screenshot of the trained model's training and validation loss plot and upload it together with your solution as `f2_2.png`.

## Exercise 3 – Multiclass Softmax Classifier

In this exercise, the task is to train and test a multiclass softmax classifier. The dataset used will be the well-known `iris` dataset.

- Implement the `get_iris_dataset` function, which loads the data using the `load_iris` function from the `sklearn.datasets` library.
    - Apply an appropriate scaler to the features.
    - Split the dataset into training and testing sets using the following proportions: 80% training, 20% testing.
    - Return `X_train`, `y_train`, `X_test`, and `y_test`.
- Implement the details of the `SoftmaxClassifier` class, which inherits from `nn.Module`, based on the features of the loaded dataset.
- Implement the `fit_softmax_classifier` function, which trains an instance of the previously created model class using the training dataset.
    - After training is complete, calculate and print the test accuracy using the test dataset.
    - Visualize the training loss curve over the epochs.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def get_iris_dataset():
  # TODO
  X = None
  y = None

  scaler = None # TODO

  # TODO: return X_train, y_train, X_test, y_test
```

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class SoftmaxClassifier(nn.Module):
  # TODO
  pass
```

```python
def fit_softmax_classifier(epochs = 100):
  model = SoftmaxClassifier()

  X_train, y_train, X_test, y_test = get_iris_dataset()

  # TODO: full implementation of the training process.

  # TODO: printing the test accuracy.
  # print(f"\nTest accuracy: X%")

  # TODO: plotting the training loss over the epochs.
```

```python
fit_softmax_classifier()
```

!!! note ":camera: Screenshot"
    Take a screenshot of the printed test accuracy and the training loss plot, and upload it together with your solution as `f3.png`.

## Exercise 4 – Handwritten Number Classifier

In this exercise, the task is to train and test a handwritten number classifier. The dataset used will be the well-known `MNIST` dataset.

- Implement the `get_mnist_dataset` function, which loads the data using the appropriate function from the `torchvision.datasets` library.
    - Apply suitable computer vision transformations.
    - Create the training and testing datasets.
    - Wrap the datasets into dataloaders.
    - Return `train_loader` and `test_loader`.
- Implement the `MNISTClassifier` class, which inherits from `nn.Module`, based on the features of the loaded dataset.
- Implement the `fit_mnist_classifier` function, which trains an instance of the previously created model class using the training dataset.
    - After training is complete, calculate and print the test accuracy using the test dataset.
    - Visualize the training loss curve over the epochs.

```python
from torchvision import datasets, transforms

def get_mnist_dataset():
  batch_size=64

  # TODO
  transform = None

  # TODO
  train_dataset = None

  # TODO
  test_dataset = None

  # TODO
  train_loader = None

  # TODO
  test_loader = None

  return train_loader, test_loader
```

```python
class MNISTClassifier(nn.Module):
  pass
```

```python
import matplotlib.pyplot as plt

def fit_mnist_classifier(epochs = 10):
  model = MNISTClassifier()

  train_loader, test_loader = get_mnist_dataset()


  # TODO: Printing the test accuracy.
  print(f"\nTest Accuracy: X%")

  # TODO: Plotting the training loss over epochs.
```

```python
fit_mnist_classifier()
```

!!! note ":camera: Screenshot"
    Take a screenshot of the printed test accuracy and the training loss plot, and upload it together with your solution as `f4.png`.

## Giving feedback

Please share your opinion about this laboratory. You can include:

- Your general impression of the laboratory.
- The things you liked or did not like.
- Whether you found the lab easy or difficult.
- Which exercises you enjoyed the most.
- What was necessary but not included in this laboratory.
- What was unnecessary but included in this laboratory.
