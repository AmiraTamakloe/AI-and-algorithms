import nn
from backend import PerceptronDataset, RegressionDataset, DigitClassificationDataset


class PerceptronModel(object):
    def __init__(self, dimensions: int) -> None:
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self) -> nn.Parameter:
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** Done: COMPLETE HERE FOR QUESTION 1 ***"
        w = self.get_weights()
        return nn.DotProduct(w, x)

    def get_prediction(self, x: nn.Constant) -> int:
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** Done: COMPLETE HERE FOR QUESTION 1 ***"
        score = self.run(x)
        return 1 if nn.as_scalar(score) >= 0 else -1


    def train(self, dataset: PerceptronDataset) -> None:
        """
        Train the perceptron until convergence.
        """
        "*** Done: COMPLETE HERE FOR QUESTION 1 ***"
        batch_size = 1
        need_update = True
        while need_update:
            need_update = False
            for x, y in dataset.iterate_once(batch_size):
                if self.get_prediction(x) != nn.as_scalar(y):
                    self.w.update(x, nn.as_scalar(y))
                    need_update = True


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: DONE HERE FOR QUESTION 2 ***"
        hidden_layer_size = 200
        self.w1 = nn.Parameter(1, hidden_layer_size)
        self.b1 = nn.Parameter(1, hidden_layer_size)
        self.w2 = nn.Parameter(hidden_layer_size, hidden_layer_size)
        self.b2 = nn.Parameter(1, hidden_layer_size)
        self.w3 = nn.Parameter(hidden_layer_size, 1)
        self.b3 = nn.Parameter(1, 1)


    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** TODO: DONE HERE FOR QUESTION 2 ***"
        layer1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        layer2 = nn.ReLU(nn.AddBias(nn.Linear(layer1, self.w2), self.b2))
        y_pred = nn.AddBias(nn.Linear(layer2, self.w3), self.b3)
        return y_pred

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** TODO: DONE HERE FOR QUESTION 2 ***"
        y_pred = self.run(x)
        loss = nn.SquareLoss(y_pred, y)
        return loss


    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: DONE HERE FOR QUESTION 2 ***"
        batch_size = 1
        learning_rate = 0.05
        threshold = 0.02
        need_update = True

        while need_update:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])

                self.w1.update(gradients[0], -learning_rate)
                self.b1.update(gradients[1], -learning_rate)
                self.w2.update(gradients[2], -learning_rate)
                self.b2.update(gradients[3], -learning_rate)
                self.w3.update(gradients[4], -learning_rate)
                self.b3.update(gradients[5], -learning_rate)
            
            final_loss = nn.as_scalar(loss)
            if final_loss <= threshold:
                need_update = False


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
