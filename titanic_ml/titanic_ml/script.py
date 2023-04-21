# imports

import pandas as pd
import torch
import torch.nn as nn
import torch.utils
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor

# classes

# define model class


class TitanicModel(nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        # Define layers

        # nn.Linear() preforms a linear transformation on data,
        # which is equivalent to multiplying the input by weight matrix and adding a bias vector.
        # the arguments are input size and output size.
        # Input: we have 6 features on input, so it's 6 on the input
        # Output: number 32 is arbitrary and can be changed according to the problem and the data.
        # Generally, having more hidden units can increase the expressive power of the network, but
        # also increase the risk of overfitting and the computational cost
        self.linear1 = nn.Linear(6, 32)

        # A ReLU activation function applies a non-linear transformation on the
        # input, which is mathematically equivalent to replacing all negative values
        # with zero. The ReLU function helps to introduce non-linearity in the network
        # and avoid the problem of vanishing gradients
        self.relu1 = nn.ReLU()

        # A dropout layer randomly drops out some of the units in the previous
        # layer with a certain probability, which is specified by the argument
        # of nn.Dropout(). In this case, the probability is 0.2, which means that 20% of
        # the units will be dropped out. The dropout layer helps to prevent overfitting and
        # improve generalization by reducing the co-dependence of the units
        self.dropout1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(16, 1)

        # We stopped at linear3 because we have three linear layers in our network.
        # We can have more or less linear layers depending on the network architecture we choose.
        # There is no fixed rule for how many layers to use, but it depends on the complexity
        # and the size of the data and the problem. Generally, having more layers can increase
        # the depth and the abstraction level of the network, but also increase the difficulty of
        # training and the possibility of vanishing or exploding gradients

        # nn.Sigmoid() defines a sigmoid activation function, which is another type of non-linear
        # transformation that can be applied to the input. A sigmoid function maps any real number
        # to a value between 0 and 1, which can be interpreted as a probability or a confidence score
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Define forward pass
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = self.sigmoid(x)

        return x

# functions


def prepare_datasets(train: pd.DataFrame, test: pd.DataFrame):
    train['Pclass'].fillna(train['Pclass'].mean(), inplace=True)
    test['Pclass'].fillna(test['Pclass'].mean(), inplace=True)
    train['Age'].fillna(train['Age'].mean(), inplace=True)
    test['Age'].fillna(test['Age'].mean(), inplace=True)
    train['SibSp'].fillna(train['SibSp'].mean(), inplace=True)
    test['SibSp'].fillna(test['SibSp'].mean(), inplace=True)
    train['Parch'].fillna(train['Parch'].mean(), inplace=True)
    test['Parch'].fillna(test['Parch'].mean(), inplace=True)
    train['Fare'].fillna(train['Fare'].mean(), inplace=True)
    test['Fare'].fillna(test['Fare'].mean(), inplace=True)

    # Convert categorical data
    d = {'male': 0, 'female': 1}
    train['Sex'] = train['Sex'].map(d)
    test['Sex'] = test['Sex'].map(d)

    return (train, test)


def random_forest(train_path: str, test_path: str, out_name: str = "sklearn_submission.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train, test = prepare_datasets(train, test)

    train_labels = train['Survived']
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    train_features = pd.get_dummies(train[features])
    test_features = pd.get_dummies(test[features])

    model = RandomForestRegressor(n_estimators=1000, random_state=1, n_jobs=-1)

    model.fit(train_features, train_labels)

    pred = model.predict(test_features).round().astype(int)

    output = pd.DataFrame(
        {'PassengerId': test['PassengerId'], 'Survived': pred})
    output.to_csv(out_name, index=False)


def torch_predict(train_path: str, test_path: str, out_name: str = "torch_submission.csv"):
    # Load data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Drop bad values
    train, test = prepare_datasets(train, test)

    # Select features and target
    X_train = train[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].values
    y_train = train["Survived"].values
    X_test = test[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].values

    # Convert to tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()

    # Create model instance
    model = TitanicModel()

    # A loss function is a function that measures how well the model is performing on the data.
    # It compares the output of the model with the true target and calculates a numerical value
    # that represents the error or the discrepancy between them. The goal of the training process
    # is to minimize the loss function by adjusting the model parameters. We selected BCELoss as the
    # loss function because it is suitable for binary classification problems. BCELoss stands for
    # binary cross-entropy loss, which is a common way of quantifying the difference between two
    # probability distributions, such as the model output and the true target
    loss_function = nn.BCELoss()

    # An optimizer is an algorithm that updates the model parameters based on the gradients of the loss
    # function. It determines how fast and in what direction the model parameters should change to reduce
    # the loss. We selected Adam as the optimizer because it is a popular and efficient optimizer that
    # adapts the learning rate for each parameter and uses momentum to accelerate the convergence.
    # Adam stands for adaptive moment estimation, which is a combination of two other optimizers:
    # RMSProp and Momentum
    optimizer = optim.Adam(model.parameters())

    # The number of epochs is a hyperparameter that controls how many times the model sees the
    # entire training data. There is no definitive answer for how to choose the number of
    # epochs, but it depends on the size and the complexity of the data and the problem.
    # Generally, having more epochs can improve the model performance, but also increase the
    # risk of overfitting and the training time. A common way of finding a good number of
    # epochs is to use early stopping, which is a technique that stops the training when the
    # validation loss stops improving or starts increasing
    epochs = 10

    # The batch size is another hyperparameter that controls how many samples are fed to the
    # model at each iteration. There is no definitive answer for how to choose the batch size,
    # but it depends on the memory and the computational resources available, as well as the
    # characteristics of the data and the problem. Generally, having a larger batch size can
    # reduce the noise and the variance in the gradient estimation, but also reduce the exploration
    # and the generalization ability of the model. A common way of finding a good batch size is to
    # use a trade-off between speed and accuracy, or to use a dynamic batch size that
    # changes during the training
    batch_size = 64

    # same as loss = 0
    loss = torch.tensor(0)

    # Define training loop
    for epoch in range(epochs):
        # We shuffle the data to avoid any bias or correlation in the order of the samples.
        # If the data is not shuffled, the model might learn some patterns or features that
        # are specific to the order of the data, which might not generalize well to new data.
        # Shuffling the data helps to make the training process more robust and unbiased.
        permutation = torch.randperm(X_train.size()[0])

        X_train = X_train[permutation]
        y_train = y_train[permutation]

        # Iterate over batches
        for i in range(0, X_train.size()[0], batch_size):
            # Get batch of features and target
            batch_x = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            # A gradient is a vector that contains the partial derivatives of the loss function
            # with respect to each model parameter. A gradient tells us how much the loss
            # function will change if we change each parameter by a small amount

            # Clear the gradients of the model parameters from the previous iteration.
            # This is necessary because PyTorch accumulates the gradients by default, which
            # means that if we don’t clear them, they will be added to the new gradients,
            # resulting in incorrect updates. By clearing the gradients, we ensure that we only
            # use the current gradients for updating the model parameters
            optimizer.zero_grad()

            # Feed batch to model and get output
            output = model(batch_x)

            # Calculate loss
            loss = loss_function(output, batch_y.unsqueeze(1))

            # We compute the gradients of the loss function with respect to the model parameters
            # using the chain rule of differentiation. This is necessary because we need to know
            # how much and in what direction each parameter should change to reduce the loss.
            # By computing the gradients, we obtain a vector of numbers that represents the direction
            # and magnitude of the error for each parameter
            loss.backward()

            # We update the model parameters using the gradients and the optimizer algorithm.
            # This is necessary because we want to adjust the model parameters to minimize the
            # loss function. By updating the parameters, we move them slightly towards the optimal
            # values that produce the lowest loss
            optimizer.step()

        # Print loss value for each epoch
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Evaluate model on test data
    predictions = model(X_test)

    # Round predictions to 0 or 1
    predictions = torch.round(predictions)

    # Convert predictions to numpy array of ints
    y_pred = predictions.detach().numpy().astype(int)

    # Create new dataframe with PassengerId and Survived columns
    submission = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Survived": y_pred.flatten()})

    # Export dataframe to csv file
    submission.to_csv(out_name, index=False)


def main():
    train_path = "../data/titanic/train.csv"
    test_path = "../data/titanic/test.csv"

    random_forest(train_path, test_path)

    torch_predict(train_path, test_path)

# execute


main()
