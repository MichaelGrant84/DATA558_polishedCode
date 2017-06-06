# DATA558_polishedCode
## l2 - regularized logistic regression
## Michael Grant

I polished my logistic regression using the l2 regularized penalty code from homework 3. 

This code took advantage of the fast-gradient descent algorithm, the initial eta set using the Lippshitz function and then optimizing the step size using the backtracking rule.

The code should run as is. The dataset used was the spam dataset and is downloaded directly via an embedded URL in the py file. The function then outputs the final set of beta coefficients, the accuracy of the prediction, the predicted values and the confusion matrix. Additionaly a plot showing the objective function value by iteration should show as a popup. 

The synthetic data is simply random numbers with different means labeled either 0 or 1. This data is generated within the code.

The required packages in order for this code to run properly are:

copy
matplotlib.pyplot
numpy
pandas
scipy.linalg
sklearn
sklearn.linear_model
sklearn.metrics
sklearn.model_selection

All packages are automatically imported in the code, but they will need to be pip or conda installed beforehand. 
