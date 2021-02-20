"""
CSCC11 - Introduction to Machine Learning, Winter 2021, Assignment 1
B. Chan, S. Wei, D. Fleet
"""

import numpy as np

class PolynomialRegression:
    def __init__(self, K):
        """ This class represents a polynomial regression model.

        TODO: You will need to implement the methods of this class:
        - predict(X): ndarray -> ndarray
        - fit(train_X, train_Y): ndarray, ndarray -> None
        - fit_with_l2_regularization(train_X, train_Y, l2_coef): ndarray, ndarray, float -> None
        Implementation description will be provided under each method.
        
        NOTE: The bias term is the first element in self.parameters (i.e. self.parameters = [w_0, w_1, ..., w_K]^T).

        Args:
        - K (int): The degree of the polynomial. Note: 1 <= K <= 10

        ASIDE: Can you see how we can abstract this code?
        """
        assert 1 <= K <= 10, f"K must be between 1 and 10. Got: {K}"
        self.K = K

        # Remember that we have K weights and 1 bias.
        self.parameters = np.ones((K + 1, 1), dtype=np.float)

    def predict(self, X):
        """ This method predicts the output of the given input data using the model parameters.
        Recall that a polynomial function is defined as:

        Given a single scalar input x,
        f(x) = w_0 + w_1 * x + w_2 * x^2 + ... + w_K * x^K

        TODO: You will need to implement the above function and handle multiple scalar inputs,
              formatted as a N-column vector.
        
        NOTE: You must not iterate through inputs.
        
        Args:
        - X (ndarray (shape: (N, 1))): A N-column vector consisting N scalar input data.

        Output:
        - ndarray (shape: (N, 1)): A N-column vector consisting N scalar output data.
        """
        assert X.shape == (X.shape[0], 1)
        
        # ====================================================
        # TODO: Implement your solution within the box
        
        powers = np.arange(0, self.K+1) # array of powers from 0 to K 
        #print(powers)
        return (X ** powers) @ self.parameters # matrix mult; return Nx1 array of w_k*x^k
    
        # ====================================================

    def fit(self, train_X, train_Y):
        """ This method fits the model parameters, given the training inputs and outputs.

        Recall that the optimal parameters are:
        parameters = (X^{T}X)^{-1}X^{T}Y

        TODO: You will need to replace self.parameters to the optimal parameters. Remember that the shape
              of the self.parameters is (K + 1, 1), where the first entry is the bias

        NOTE: Do not forget that we are using polynomial basis functions!

        Args:
        - train_X (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training inputs.
        - train_Y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training outputs.
        """
        assert train_X.shape == train_Y.shape and train_X.shape == (train_X.shape[0], 1), f"input and/or output has incorrect shape (train_X: {train_X.shape}, train_Y: {train_Y.shape})."
        assert train_X.shape[0] >= self.K, f"require more data points to fit a polynomial (train_X: {train_X.shape}, K: {self.K}). Do you know why?"

        # ====================================================
        # TODO: Implement your solution within the box
        
        powers = np.arange(0,self.K + 1) # array of powers from 0 to K
        x = train_X ** powers # taking x training data and plugging it into function
        invMat = np.linalg.inv(x.T @ x) # (x^Tx)^-1; reference the linear_regression doc
        xyMat = x.T @ train_Y # (x^Ty)
        w = invMat @ xyMat # (x^Tx)^-1 * (x^Ty)
        self.parameters = w # set parameters to optimal parameters
       # return self.parameters 
        
        # ====================================================

        assert self.parameters.shape == (self.K + 1, 1)

    def fit_with_l2_regularization(self, train_X, train_Y, l2_coef):
        """ This method fits the model parameters with L2 regularization, given the training inputs and outputs.

        Recall that the optimal parameters are:
        parameters = (X^{T}X + lambda*I)^{-1}X^{T}Y

        TODO: You will need to replace self.parameters to the optimal parameters. Remember that the shape
              of the self.parameters is (K + 1, 1), where the first entry is the bias

        NOTE: Do not forget that we are using polynomial basis functions!

        Args:
        - train_X (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training inputs.
        - train_Y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training outputs.
        - l2_coef (float): The lambda term that decides how much regularization we want.
        """
        assert train_X.shape == train_Y.shape and train_X.shape == (train_X.shape[0], 1), f"input and/or output has incorrect shape (train_X: {train_X.shape}, train_Y: {train_Y.shape})."

        # ====================================================
        # TODO: Implement your solution within the box

        powers = np.arange(0,self.K + 1) # array of powers from 0 to K
        x = train_X ** powers # taking x training data and plugging it into function
        idMat = np.eye(self.K + 1) # creating identity matrix of size KxK
        invMat = np.linalg.inv((x.T @ x) + l2_coef*idMat) # (x^Tx + lambda*I)^-1
        xyMat = x.T @ train_Y # (x^Ty)
        w = invMat @ xyMat # (x^Tx + lambda*I)^-1 * (x^Ty)
        self.parameters = w # set parameters to optimal parameters
       
        #return self.parameters 
        
        # ====================================================

        assert self.parameters.shape == (self.K + 1, 1)

    def compute_mse(self, X, observed_Y):
        """ This method computes the mean squared error.

        Args:
        - X (ndarray (shape: (N, 1))): A N-column vector consisting N scalar inputs.
        - observed_Y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar observed outputs.

        Output:
        - (float): The mean squared error between the predicted Y and the observed Y.
        """
        assert X.shape == observed_Y.shape and X.shape == (X.shape[0], 1), f"input and/or output has incorrect shape (X: {X.shape}, observed_Y: {observed_Y.shape})."
        pred_Y = self.predict(X)
        return ((pred_Y - observed_Y) ** 2).mean()


if __name__ == "__main__":
    # You can use linear regression to check whether your implementation is correct.
    # NOTE: This is just a quick check but does not cover all cases.
    model = PolynomialRegression(K=1)
    train_X = np.expand_dims(np.arange(10), axis=1)
    train_Y = np.expand_dims(np.arange(10), axis=1)

    model.fit(train_X, train_Y)
    optimal_parameters = np.array([[0.], [1.]])
    print("Correct optimal weights: {}".format(np.allclose(model.parameters, optimal_parameters)))

    pred_Y = model.predict(train_X)
    print("Correct predictions: {}".format(np.allclose(pred_Y, train_Y)))

    # Change parameters to suboptimal for next test
    model.parameters += 1
    model.fit_with_l2_regularization(train_X, train_Y, l2_coef=0)
    print("Correct optimal weights: {}".format(np.allclose(model.parameters, optimal_parameters)))

    # Regularization pulls the weights closer to 0.
    optimal_parameters = np.array([[0.0231303], [0.99460293]])
    model.fit_with_l2_regularization(train_X, train_Y, l2_coef=0.5)
    print("Correct optimal weights: {}".format(np.allclose(model.parameters, optimal_parameters)))
