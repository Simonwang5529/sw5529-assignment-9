import numpy as np
import matplotlib
matplotlib.use('Agg')  # Switch backend to 'Agg' to generate images without a GUI
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch

# Define activation functions
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2
# Define ReLU and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

        # Assign activation functions
        if activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.activation(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y):
        m = X.shape[0]
        dZ2 = self.A2 - y
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.activation_derivative(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(int).reshape(-1, 1)
    return X, y

def plot_hidden_space(ax, hidden_features, y):
    ax.clear()
    ax.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
    # Add a translucent surface plot over the scatter points
    x, y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    z = x**2 - y**2 
    ax.plot_surface(x, y, z, alpha=0.2, color='gray')
    ax.set_title("Hidden Space at Step")
    ax.set_xlabel("Hidden 1")
    ax.set_ylabel("Hidden 2")
    ax.set_zlabel("Hidden 3")

def plot_input_space(ax, X, y, mlp):
    ax.clear()
    x1, x2 = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    grid = np.c_[x1.ravel(), x2.ravel()]
    preds = mlp.forward(grid).reshape(x1.shape)
    ax.contourf(x1, x2, preds, levels=50, cmap='bwr', alpha=0.5)
    ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax.set_title("Input Space at Step")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
def plot_gradients(ax, W1, W2):
    ax.clear()
    ax.set_title("Gradients at Step")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 2.5)
    ax.axis("off")

    # Draw nodes
    input_nodes = [0, 1]  # x1, x2
    hidden_nodes = [2, 3, 4]  # h1, h2, h3
    output_node = [5]  # y

    # Draw input layer nodes
    for i in input_nodes:
        ax.text(i, 0, f"x{i+1}", ha='center', va='center', fontsize=10, color='blue')

    # Draw hidden layer nodes
    for i, h in enumerate(hidden_nodes):
        ax.text(i, 1, f"h{i+1}", ha='center', va='center', fontsize=10, color='green')

    # Draw output layer node
    ax.text(1, 2, "y", ha='center', va='center', fontsize=10, color='red')

    # Draw edges between input and hidden layer
    for i in input_nodes:  # x1, x2
        for j, h in enumerate(hidden_nodes):  # h1, h2, h3
            weight = W1[i, j]
            ax.plot([i, j], [0, 1], linewidth=2 + 10 * np.abs(weight), color='purple', alpha=0.7)

    # Draw edges between hidden and output layer
    for j, h in enumerate(hidden_nodes):  # h1, h2, h3
        weight = W2[j, 0]  # Only one output node (y), so W2[j, 0]
        ax.plot([j, 1], [1, 2], linewidth=2 + 10 * np.abs(weight), color='orange', alpha=0.7)
def update(frame, mlp, X, y, ax_input, ax_hidden, ax_gradient):
    for _ in range(10):  # Perform multiple steps to see clear changes
        mlp.forward(X)
        mlp.backward(X, y)

    hidden_features = mlp.A1
    plot_hidden_space(ax_hidden, hidden_features, y)
    plot_input_space(ax_input, X, y, mlp)
    plot_gradients(ax_gradient, mlp.W1, mlp.W2)  # Pass both W1 and W2

def visualize(activation='tanh', lr=0.1, step_num=1000):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    fig = plt.figure(figsize=(18, 6))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    ani = FuncAnimation(fig, partial(update, mlp=mlp, X=X, y=y, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient), 
                        frames=step_num // 10, repeat=False)
    ani.save("results/visualize.gif", writer="pillow", fps=10)
    plt.close()

if __name__ == "__main__":
    visualize(activation="relu", lr=0.1, step_num=1000)