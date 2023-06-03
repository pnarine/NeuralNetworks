import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
class DenseLayer:
    def __init__(self, input_size, output_size, activation='None'):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros(output_size)
        self.activation = activation

    # activation functions and their derivatives
    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        x = np.clip(x, -100, 100)
        return 1 / (1 + np.exp(-x))

    def deriv_relu(self, x):
        return 1 * (x > 0)

    def deriv_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # depending on the value of self.activation, activation_func and deriv_activation functions return
    # activation function and its derivative, if activation is None activation_func returns the same value,
    # and derivative returns identic 1 to have no effect
    def activation_func(self, x):
        if self.activation == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation == 'relu':
            return self.relu(x)
        else:
            return x

    def deriv_activation(self, x):
        if self.activation == 'sigmoid':
            return self.deriv_sigmoid(x)
        elif self.activation == 'relu':
            return self.deriv_relu(x)
        else:
            return x - (x - 1)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.activation_func(np.dot(inputs, self.weights) + self.biases)
        return self.output

    def backward(self, grad_output, learning_rate):

        activation_derivative = self.deriv_activation(np.dot(self.inputs, self.weights) + self.biases)
        grad_activation = grad_output * activation_derivative

        grad_weights = np.dot(self.inputs.T, grad_activation)
        grad_biases = np.sum(grad_activation, axis=0)

        grad_input = np.dot(grad_activation, self.weights.T)

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input


class DenseNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)

    def train(self, x_train, y_train, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            outputs = self.forward(x_train)
            loss = np.mean((outputs - y_train) ** 2)
            grad_output = 2 * (outputs - y_train) / len(x_train)
            self.backward(grad_output, learning_rate)


    def predict(self, x_test):
        return self.forward(x_test)


def main():

    # Generate synthetic dataset
    X, y = make_regression(n_samples=100, n_features=10, noise=0.5, random_state=42)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Standardize the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Y_train_scaled = scaler.fit_transform(Y_train)

    # Train scikit-learn's LinearRegression model
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)

    # Predict with scikit-learn's LinearRegression model
    y_pred_lr = lr_model.predict(X_test_scaled)

    # Train the DenseNetwork implemented from scratch
    dense_net = DenseNetwork()
    dense_net.add_layer(DenseLayer(10, 10))
    dense_net.add_layer(DenseLayer(10, 1))

    learning_rate = 0.001
    num_epochs = 1000
    dense_net.train(X_train_scaled, y_train, num_epochs, learning_rate)
    # Predict with the DenseNetwork
    y_pred_dense = dense_net.predict(X_test_scaled)

    # Compare the results
    print("Mean Squared Error (sklearn LinearRegression):", mean_squared_error(y_test, y_pred_lr))
    print("Mean Squared Error (DenseNetwork implemented from scratch):", mean_squared_error(y_test, y_pred_dense))


if __name__ == '__main__':
    main()