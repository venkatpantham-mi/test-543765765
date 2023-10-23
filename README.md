### **1. Linear Regression and Normal Equations**


*Question*: Explain how the normal equations and linear algebra are used to solve linear regression problems.

*Solution*: 
Linear regression aims to find a linear relationship between input features and a target variable. The normal equations are a method for finding the coefficients of the linear regression model.

Given a dataset with input features X (a matrix) and target values y (a vector), the linear regression model can be represented as:

    y = X * θ

Where:
- `y` is the target variable.
- `X` is the feature matrix where each row represents an observation, and each column represents a feature.
- `θ` is the vector of coefficients (including the intercept).

To find the optimal coefficients θ, we can use the normal equations:

    X^T * X * θ = X^T * y

Where:
- `X^T` is the transpose of the feature matrix X.
- `X^T * X` is a square matrix.
- `X^T * y` is a vector.

To solve for θ, we can use the following equation:

    θ = (X^T * X)^(-1) * X^T * y

This equation can be implemented in Python using NumPy as follows:

Keep in mind that while matrix inversion is a valid method for solving linear equations, it may not always be the most efficient or numerically stable approach for large or ill-conditioned matrices. Numerical libraries like NumPy often provide specialized functions for solving linear systems that are more stable and efficient in practice.


```python

import numpy as np

# Sample data
X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([2, 3, 4])

# Calculate the coefficients using the normal equations
X_transpose = np.transpose(X)
theta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
print("Coefficients (theta):", theta)
```

    Coefficients (theta): [1. 1.]
    

Question1: Consider a simple linear regression problem where we want to predict house prices based on the number of bedrooms. Please solve theta using the abobe example and disuss your result 


```python
import numpy as np

# Dummy data
bedrooms = np.array([1, 2, 3, 4, 5])
house_prices = np.array([200, 300, 400, 450, 500])

# Create the feature matrix X with a column of ones for the intercept
X = np.column_stack((np.ones_like(bedrooms), bedrooms))

# Calculate the coefficients using the normal equations
X_transpose = X.T
theta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(house_prices)

print("Coefficients (theta):", theta)

```

    Coefficients (theta): [145.  75.]
    


```python

```


```python

```

### **2.Matrix Inversion and Linear Equations**



Question: Discuss the use of matrix inversion to solve systems of linear equations.

Solution:
Matrix inversion is a technique used to solve systems of linear equations of the form Ax = b, where A is a square matrix, x is the vector of unknowns, and b is the right-hand side vector.

The solution to the system of equations can be found using matrix inversion as follows:

1. Given the equation Ax = b, we want to solve for x.
2. Multiply both sides of the equation by the inverse of matrix A: A^(-1).
3. This gives us A^(-1) * (Ax) = A^(-1) * b.
4. Since A^(-1) * A is the identity matrix (I), we have Ix = A^(-1) * b.
5. Since Ix is simply x, we have x = A^(-1) * b.

 



 In data science, how can matrix inversion be applied to solve a system of linear equations? Provide a basic example using Python and NumPy.
 
Matrix inversion is a technique used in data science to solve systems of linear equations efficiently. This is commonly applied in various fields, including statistics, machine learning, and optimization. Matrix inversion is used when the coefficient matrix is invertible (i.e., its determinant is non-zero).


Let's consider a simple example of solving a system of linear equations using matrix inversion:

```python
import numpy as np

# Coefficient matrix A
A = np.array([[2, 3],
              [4, 5]])

# Right-hand side vector b
b = np.array([8, 10])

# Check if the matrix A is invertible (non-singular)
if np.linalg.det(A) != 0:
    # Solve for x using matrix inversion
    x = np.linalg.inv(A).dot(b)
    print("Solution (x, y):", x)
else:
    print("Matrix A is singular (not invertible).")
```

In this example:

1. We define the coefficient matrix `A` representing the coefficients of `x` and `y` in a system of linear equations.

2. We define the right-hand side vector `b` containing the constants on the right-hand side of the equations.

3. We check if the matrix `A` is invertible by verifying that its determinant is non-zero. If the determinant is zero, the matrix is singular and cannot be inverted.

4. If `A` is invertible, we use NumPy to calculate the solution vector `x` by performing matrix inversion and then multiplying it by `b`.

5. We print the values of `x` and `y`, which represent the solutions to the system of linear equations.

This example demonstrates how matrix inversion can be applied to solve a basic system of linear equations, which is a fundamental concept in data science and mathematics.


```python
#Python example using NumPy to solve a system of linear equations using matrix inversion:

import numpy as np

# Coefficient matrix A
A = np.array([[2, 3], [4, 5]])

# Right-hand side vector b
b = np.array([7, 10])

# Solve for x using matrix inversion
x = np.linalg.inv(A).dot(b)

print("Solution (x):", x)
```


Question2:  Given the following system of linear equations:


```python

```
2x + 3y = 8
4x - y  = 7
```
Use Python and NumPy to find the values of `x` and `y` that satisfy these equations.


```


```python
A = np.array([[2, 3],
              [4, -1]])
B = np.array([8, 7])
X = np.linalg.solve(A, B)
x, y = X
```

Describe how matrix inversion can be used to solve the linear regression problem, and discuss any limitations or assumptions associated with this approach.

## 3. Linear Regression with Matrix Inversion

Question3:  You are working on a data science project where you need to perform linear regression to predict a target variable 'y' based on multiple features 'X1', 'X2', and 'X3'. Your goal is to use matrix inversion to find the coefficients ('theta') of the linear regression model. The dataset dummy_data.csv contains the following data:


```python
X1, X2, X3, y
1, 2, 3, 12
2, 3, 4, 18
3, 4, 5, 24
4, 5, 6, 30
5, 6, 7, 36

```

Write Python code to perform the following tasks:

- Load the dataset from dummy_data.csv into a Pandas DataFrame.
- Create a feature matrix 'X' by selecting the columns 'X1', 'X2', and 'X3'.
- Create a target vector 'y' by selecting the 'y' column.
- Add a column of ones to the feature matrix 'X' to represent the intercept term.
- Use matrix inversion to calculate the coefficients 'theta' of the linear regression model.
- Print the coefficients 'theta'.


```python
import pandas as pd
import numpy as np

# 1. Load the dataset into a DataFrame
df = pd.read_csv('dummy_data.csv')

# 2. Separate the features and the target variable
X = df[['X1','X2','X3']].values
y = df[' y'].values

# 3. Add a column of ones for the intercept
X = np.column_stack((np.ones(X.shape[0]), X))

# 4. Use matrix inversion and .dot() to calculate the coefficients
# Checking if the matrix is singular or near-singular
det = np.linalg.det(X.T.dot(X))
if np.isclose(det, 0):
    # Using the pseudoinverse for near-singular matrices
    pseudo_inv = np.linalg.pinv(X.T.dot(X))
    theta = pseudo_inv.dot(X.T).dot(y)
else:
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# 5. Print the coefficients
print("Coefficients (theta):", theta)
```

## 4: Matrix Inversion and Eigenvlaues


**Question 4: Matrix Inversion and Ill-Conditioned Matrices**

Explain the concept of ill-conditioned matrices in the context of matrix inversion. Provide an example of an ill-conditioned matrix and discuss why matrix inversion can lead to numerical instability for such matrices.


Ill-conditioned matrices have determinants near zero, making their inversion numerically unstable. Small input changes yield large output differences.

**Question 5: Eigenvalues and Eigenvectors**

 In data science, why are eigenvalues and eigenvectors important? Provide an example of how they can be applied in a practical data analysis scenario.




Eigenvalues and eigenvectors provide insight into the nature and behavior of matrices. An eigenvector of a matrix A is a vector represented by a matrix X such that when A is multiplied by X, then the direction of X doesn't change, though its scale may change. This scalar is the eigenvalue (λ) associated with the eigenvector.

## Bonus Questions

Question: Explain what a covariance matrix is in the context of linear algebra. How is it calculated, and what does it represent in data analysis?


```python

```

Question: What is a correlation matrix, and how does it relate to the covariance matrix? How is it calculated, and why is it often preferred in data analysis?


```python

```

Question: Provide examples of how covariance and correlation matrices are used in data science and statistical analysis. How can they help in understanding data relationships and making decisions?


```python

```

Question: Write Python code to calculate the covariance between two sets of data, data1 and data2. Assume that both data1 and data2 are NumPy arrays of the same length.


```python

```


```python
import numpy as np

# Sample data
data1 = np.array([1, 2, 3, 4, 5])
data2 = np.array([5, 4, 3, 2, 1])

# Calculate the covariance
------

print("Covariance:", covariance)

```


```python

```

Question: Write Python code to calculate the Pearson correlation coefficient (correlation) between two sets of data, data1 and data2. Assume that both data1 and data2 are NumPy arrays of the same length.


```python
import numpy as np

# Sample data
data1 = np.array([1, 2, 3, 4, 5])
data2 = np.array([5, 4, 3, 2, 1])

# Calculate the correlation coefficient


print("Correlation Coefficient:", correlation)

```


```python

```
