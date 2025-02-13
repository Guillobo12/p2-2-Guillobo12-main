# Laboratory practice 2.2: KNN classification
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
import numpy as np  
import seaborn as sns


def minkowski_distance(a, b, p=2):
    """
    Compute the Minkowski distance between two arrays.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.
        p (int, optional): The degree of the Minkowski distance. Defaults to 2 (Euclidean distance).

    Returns:
        float: Minkowski distance between arrays a and b.
    """

    distancia = 0
    for i in range (0,len(b)):
        distancia += abs(b[i]-a[i])**p
    distancia = distancia**(1/p)
    return distancia


# k-Nearest Neighbors Model

# - [K-Nearest Neighbours](https://scikit-learn.org/stable/modules/neighbors.html#classification)
# - [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)


class knn:
    def __init__(self):
        self.k = None
        self.p = None
        self.x_train = None
        self.y_train = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, k: int = 5, p: int = 2):
        """
        Fit the model using X as training data and y as target values.

        You should check that all the arguments shall have valid values:
            X and y have the same number of rows.
            k is a positive integer.
            p is a positive integer.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Target values.
            k (int, optional): Number of neighbors to use. Defaults to 5.
            p (int, optional): The degree of the Minkowski distance. Defaults to 2.
        """
        if k < 1 or p < 1: 
            raise ValueError("k and p must be positive integers.")
        if len(X_train) != len(y_train):
            raise ValueError("Length of X_train and y_train must be equal.")
        else:
            self.p = p
            self.k = k
            self.y_train = y_train
            self.x_train = X_train

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided data.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class labels.
        """
        etiqueta_moda = []
        for entidad in X:
            distancias = self.compute_distances(entidad)
            indices_knn = self.get_k_nearest_neighbors(distancias)
            etiquetas_knn = []
            for indice in indices_knn:
                etiquetas_knn.append(self.y_train[indice])
            etiquetas_knn = np.array(etiquetas_knn)
            etiqueta_moda.append(self.most_common_label(etiquetas_knn))
        etiqueta_moda = np.array(etiqueta_moda)

        return etiqueta_moda
    def predict_proba(self, X):
        """
        Predict the class probabilities for the provided data.

        Each class probability is the amount of each label from the k nearest neighbors
        divided by k.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        respuesta = []
        for entidad in X:
            distances_list = self.compute_distances(entidad)
            indices_knn = self.get_k_nearest_neighbors(distances_list)
            knn_class_dict = {}
            for indice in indices_knn:
                knn_class_dict[self.y_train[indice]] = knn_class_dict.get(self.y_train[indice],0)+1
            lista_resp = []
            for valor in knn_class_dict.values():
                lista_resp.append(valor/self.k)
            respuesta.append(lista_resp)
            if len(respuesta) == 1:
                respuesta.append(0)

        print(respuesta)
        respuesta = np.array(respuesta)
        return respuesta


    def compute_distances(self, point: np.ndarray) -> np.ndarray:
        """Compute distance from a point to every point in the training dataset

        Args:
            point (np.ndarray): data sample.

        Returns:
            np.ndarray: distance from point to each point in the training dataset.
        """
        distancias = []
        for i in range(0,len(self.x_train)):
            distancia = minkowski_distance(point,self.x_train[i])
            distancias.append(distancia)
        distancias = np.array(distancias)
        return distancias

    def get_k_nearest_neighbors(self, distances: np.ndarray) -> np.ndarray:
        """Get the k nearest neighbors indices given the distances matrix from a point.

        Args:
            distances (np.ndarray): distances matrix from a point whose neighbors want to be identified.

        Returns:
            np.ndarray: row indices from the k nearest neighbors.

        Hint:
            You might want to check the np.argsort function.
        """

        indices_knn = np.argsort(distances)
        respuesta = indices_knn[0:self.k]
        return respuesta

    def most_common_label(self, knn_labels: np.ndarray) -> int:
        """Obtain the most common label from the labels of the k nearest neighbors

        Args:
            knn_labels (np.ndarray): labels from the k nearest neighbors

        Returns:
            int: most common label
        """
        max_value = 0
        etiqueta_diccionario = {}
        for etiqueta in knn_labels:
            etiqueta_diccionario[etiqueta] = etiqueta_diccionario.get(etiqueta,0)+1
        max_value = 0
        for clase,repeticiones in etiqueta_diccionario.items():
            if repeticiones > max_value:
                max_value = repeticiones
                clase_maxima = clase

        return clase_maxima

    def __str__(self):
        """
        String representation of the kNN model.
        """
        return f"kNN model (k={self.k}, p={self.p})"



def plot_2Dmodel_predictions(X, y, model, grid_points_n):
    """
    Plot the classification results and predicted probabilities of a model on a 2D grid.

    This function creates two plots:
    1. A classification results plot showing True Positives, False Positives, False Negatives, and True Negatives.
    2. A predicted probabilities plot showing the probability predictions with level curves for each 0.1 increment.

    Args:
        X (np.ndarray): The input data, a 2D array of shape (n_samples, 2), where each row represents a sample and each column represents a feature.
        y (np.ndarray): The true labels, a 1D array of length n_samples.
        model (classifier): A trained classification model with 'predict' and 'predict_proba' methods. The model should be compatible with the input data 'X'.
        grid_points_n (int): The number of points in the grid along each axis. This determines the resolution of the plots.

    Returns:
        None: This function does not return any value. It displays two plots.

    Note:
        - This function assumes binary classification and that the model's 'predict_proba' method returns probabilities for the positive class in the second column.
    """
    # Map string labels to numeric
    unique_labels = np.unique(y)
    num_to_label = {i: label for i, label in enumerate(unique_labels)}

    # Predict on input data
    preds = model.predict(X)

    # Determine TP, FP, FN, TN
    tp = (y == unique_labels[1]) & (preds == unique_labels[1])
    fp = (y == unique_labels[0]) & (preds == unique_labels[1])
    fn = (y == unique_labels[1]) & (preds == unique_labels[0])
    tn = (y == unique_labels[0]) & (preds == unique_labels[0])

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Classification Results Plot
    ax[0].scatter(X[tp, 0], X[tp, 1], color="green", label=f"True {num_to_label[1]}")
    ax[0].scatter(X[fp, 0], X[fp, 1], color="red", label=f"False {num_to_label[1]}")
    ax[0].scatter(X[fn, 0], X[fn, 1], color="blue", label=f"False {num_to_label[0]}")
    ax[0].scatter(X[tn, 0], X[tn, 1], color="orange", label=f"True {num_to_label[0]}")
    ax[0].set_title("Classification Results")
    ax[0].legend()

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_points_n),
        np.linspace(y_min, y_max, grid_points_n),
    )

    # # Predict on mesh grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    # Use Seaborn for the scatter plot
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1", ax=ax[1])
    ax[1].set_title("Classes and Estimated Probability Contour Lines")

    # Plot contour lines for probabilities
    cnt = ax[1].contour(xx, yy, probs, levels=np.arange(0, 1.1, 0.1), colors="black")
    ax[1].clabel(cnt, inline=True, fontsize=8)

    # Show the plot
    plt.tight_layout()
    plt.show()



def evaluate_classification_metrics(y_true, y_pred, positive_label):
    """
    Calculate various evaluation metrics for a classification model.

    Args:
        y_true (array-like): True labels of the data.
        positive_label: The label considered as the positive class.
        y_pred (array-like): Predicted labels by the model.

    Returns:
        dict: A dictionary containing various evaluation metrics.

    Metrics Calculated:
        - Confusion Matrix: [TN, FP, FN, TP]
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Precision: TP / (TP + FP)
        - Recall (Sensitivity): TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    """
    #True positive
    TP = 0
    #False positive
    FP = 0
    #False negative
    FN = 0
    #True negative
    TN = 0

    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    y_pred_mapped = np.array([1 if label == positive_label else 0 for label in y_pred])


    for i in range(len(y_true_mapped)):
        true_value = y_true_mapped[i]
        predicted_value = y_pred_mapped[i]
        if predicted_value == 1:
            if true_value == 0:
                FP += 1
            else:
                TP += 1
        else:
            if true_value == 1:
                FN += 1
            else:
                TN += 1


    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Precision
    if TP + FP != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0


    # Recall (Sensitivity)
    if TP+FN != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0


    # Specificity
    if  TN + FP != 0:
        specificity = TN / (TN + FP)
    else:
        specificity = 0

    # F1 Score
    if (precision + recall) != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0



    return {
        "Confusion Matrix": [TN, FP, FN, TP],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": f1,
    }



def plot_calibration_curve(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot a calibration curve to evaluate the accuracy of predicted probabilities.

    This function creates a plot that compares the mean predicted probabilities
    in each bin with the fraction of positives (true outcomes) in that bin.
    This helps assess how well the probabilities are calibrated.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class (positive_label).
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label that is considered the positive class.
                                    This is used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins to use for grouping predicted probabilities.
                                Defaults to 10. Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "bin_centers": Array of the center values of each bin.
            - "true_proportions": Array of the fraction of positives in each bin

    """
    #lista_bins tendrá la información de cada bin
    lista_intervalos = []
    espacio_entre_bins = 1/n_bins

    for i in range(0,n_bins):
        lista_intervalos.append([0,0,0])
    
    for i in range(0,len(y_true)):
        prob_clase_elegida = y_probs[i]
        ind_bin = prob_clase_elegida//espacio_entre_bins
        ind_bin = int(ind_bin)
        lista_intervalos[ind_bin][2] += prob_clase_elegida
        if y_true[i] == positive_label:
            lista_intervalos[ind_bin][0] += 1
        lista_intervalos[ind_bin][1] += 1

    eje_x = []
    eje_y = []
    for intervalo in lista_intervalos:
        eje_x.append(intervalo[2]/intervalo[1])
        eje_y.append(intervalo[0]/intervalo[1])      
    centro_de_bins = []
    for i in range(0,n_bins):
        centro_de_bins.append((espacio_entre_bins/2+i*espacio_entre_bins))
    true_proportions = np.array(eje_y)
    print(centro_de_bins)
    
    return {"bin_centers": centro_de_bins, "true_proportions": true_proportions}



def plot_probability_histograms(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot probability histograms for the positive and negative classes separately.

    This function creates two histograms showing the distribution of predicted
    probabilities for each class. This helps in understanding how the model
    differentiates between the classes.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins for the histograms. Defaults to 10. 
                                Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "array_passed_to_histogram_of_positive_class": 
                Array of predicted probabilities for the positive class.
            - "array_passed_to_histogram_of_negative_class": 
                Array of predicted probabilities for the negative class.

    """
    y_true_mapped = np.array(y_true) == positive_label

    return {
        "array_passed_to_histogram_of_positive_class": y_probs[y_true_mapped == 1],
        "array_passed_to_histogram_of_negative_class": y_probs[y_true_mapped == 0],
    }




def plot_roc_curve(y_true, y_probs, positive_label):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    The ROC curve is a graphical representation of the diagnostic ability of a binary
    classifier system as its discrimination threshold is varied. It plots the True Positive
    Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.

    Returns:
        dict: A dictionary containing the following:
            - "fpr": Array of False Positive Rates for each threshold.
            - "tpr": Array of True Positive Rates for each threshold.

    """
    indices_probabilidades_ordenadas = np.argsort(y_probs)
    y_true = y_true[indices_probabilidades_ordenadas]
    y_probs = y_probs[indices_probabilidades_ordenadas]
    positives= sum(np.array(y_true) == positive_label)
    negatives = sum(np.array(y_true) != positive_label)
    tpr = []
    fpr = []
    umbrales_list = np.linspace(0, 1, 11)

    for umbral in umbrales_list:
        False_positives = 0
        True_positives = 0
        for i in range(len(y_true)):
            valor_verdadero = y_true[i]
            if y_probs[i] >= umbral:
                if valor_verdadero == 0:
                    False_positives += 1
                else:
                    True_positives += 1




        tpr.append(True_positives/positives)
        fpr.append(False_positives/negatives)


    return {"fpr": np.array(fpr), "tpr": np.array(tpr)}
