from typing import Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import warnings
# We will suppress ConvergenceWarnings in this task. In practice, you should take warnings more seriously.
warnings.filterwarnings("ignore")


def reduce_dimension(X_train: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """
    :param X_train: Training data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality, which has shape (n_samples, n_components), and the PCA object
    """
    
    # TODO: Create a PCA object and fit it using X_train
    #       Transform X_train using the PCA object.
    #       Print the explained variance ratio of the PCA object.
    #       Return both the transformed data and the PCA object.
    pca = PCA(n_components, random_state=42)
    pca.fit(X_train)

    X_transformed = pca.transform(X_train)
    print(f"variance ratio: {np.sum(pca.explained_variance_ratio_) * 100:.2f}%")
    return X_transformed, pca


def train_nn(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier with different number of neurons and hidden layers.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)
    
    # TODO: Train MLPClassifier with different number of layers/neurons.
    #       Print the train accuracy, validation accuracy, and the training loss for each configuration.
    #       Return the MLPClassifier that you consider to be the best.
    
    H= [(2,), (8,), (64,), (256,), (1024,), (128, 256 ,128)]
    
    scores_val = []
    scores_train = []	
    loss = []

    for i in H:
        mlp = MLPClassifier(max_iter=100, random_state=1, solver='adam', hidden_layer_sizes=i)
        mlp.fit(X_train, y_train)
        scores_val.append(mlp.score(X_val, y_val))
        scores_train.append(mlp.score(X_train, y_train))
        loss.append(mlp.loss_)

    print(f"Accuracy val data = {scores_val}")
    print(f"Accuracy train data = {scores_train}")
    print(f"Loss = {loss}")
    best_index = np.argmax(scores_val)
    
    best_mlp = MLPClassifier(max_iter=100, random_state=1, solver='adam', hidden_layer_sizes=H[best_index])
    best_mlp.fit(X_train, y_train)

    return best_mlp


def train_nn_with_regularization(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier using regularization.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # TODO: Use the code from the `train_nn` function, but add regularization to the MLPClassifier.
    #       Again, return the MLPClassifier that you consider to be the best.

    H= [(2,), (8,), (64,), (256,), (1024,), (128, 256 ,128)]
    
    scores_val = []
    scores_train = []	
    loss = []

    args = [(0.1, False), (0.0001, True), (0.1, True)]
    
    for alpha1, early_stopping1 in args:
        for i in H:
            mlp = MLPClassifier(max_iter=100, random_state=1, solver='adam', hidden_layer_sizes=i, alpha=alpha1, early_stopping=early_stopping1)
            mlp.fit(X_train, y_train)
            scores_val.append(mlp.score(X_val, y_val))
            scores_train.append(mlp.score(X_train, y_train))
            loss.append(mlp.loss_)

        print(f"Accuracy val data = {scores_val}")
        print(f"Accuracy train data = {scores_train}")
        print(f"Loss = {loss}")
    
    best_index = np.argmax(scores_val)
    best_H =  (best_index+1)%6 - 1 
    best_arg = int(np.floor(best_index/6))
    print(f"Best H = {best_H}, best arg = {best_arg}")
    
    best_mlp = MLPClassifier(max_iter=100, random_state=1, solver='adam', hidden_layer_sizes=H[best_H], alpha=args[best_arg][0], early_stopping=args[best_arg][1])  
    best_mlp.fit(X_train, y_train)

    return best_mlp


def plot_training_loss_curve(nn: MLPClassifier) -> None:
    """
    Plot the training loss curve.

    :param nn: The trained MLPClassifier
    """
    loss = nn.loss_curve_
    plt.plot(loss)
    plt.title('Losscurve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()



    # TODO: Plot the training loss curve of the MLPClassifier. Don't forget to label the axes.


def show_confusion_matrix_and_classification_report(nn: MLPClassifier, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Plot confusion matrix and print classification report.

    :param nn: The trained MLPClassifier you want to evaluate
    :param X_test: Test features (PCA-projected)
    :param y_test: Test targets
    """
    # TODO: Use `nn` to compute predictions on `X_test`.
    #       Use `confusion_matrix` and `ConfusionMatrixDisplay` to plot the confusion matrix on the test data.
    #       Use `classification_report` to print the classification report.


def perform_grid_search(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Perform GridSearch using GridSearchCV.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The best estimator (MLPClassifier) found by GridSearchCV
    """
    # TODO: Create parameter dictionary for GridSearchCV, as specified in the assignment sheet.
    #       Create an MLPClassifier with the specified default values.
    #       Run the grid search with `cv=5` and (optionally) `verbose=4`.
    #       Print the best score (mean cross validation score) and the best parameter set.
    #       Return the best estimator found by GridSearchCV.

    return None