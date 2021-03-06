"""
As noted by Murphy [1]_ Support Vector Machines (SVMs) are not
probabilistic models. This package attempts to provide a solution in order to
develop SVM which can output the uncertainty estimate alongside prediction, and
therefore, at least to some extent,  introduce benefits of probabilistic
Machine Learning to otherwise non-probabilistic method.

In this specific case, a classification task has been selected to be solved
with a SVM, even though the same methodology can be extended to solving
regression problems. For further details regarding the data set
describing the classification task available at Torres, Ranasinghe,
Sample [2]_. One characteristic of this data set deserves to be noted;
there is a heavy-imbalance across classes with one of the classes
accounting for nearly 90% of the data-set.

The strategy to include uncertainty estimates for SVM is based on the
approach defined by Osband, Blundell, Pritzel [3]_. In addition
to the above mentioned approach which is based on boot-strapping, one more
innovation is introduced. Ensemble of SVMs which generates prediction and
uncertainty estimates, is trained on a collection of data sets which are
constructed in such way that all classes are included in exactly same
proportion. Further details regarding the influence this approach has
on the decision boundary is explained in subsequent pages.

Examples
--------
>>> import pandas as pd
... import numpy as np
... from sklearn.decomposition import PCA
... from sklearn.svm import SVC
... import uncertainty_estimate_with_svm.ucf as ue_svm

>>> # Generate balanced set.
... ue_svm.reduce_set_to_equal_distribution_of_classes(
...    features_for_training=x_train,
...     targets_for_training=y_train
... )

>>> # Fit ensemble.
... ensembles = ue_svm.generate_ensemble(
...     number_of_estimators=30,
...     features_for_training=balanced_x_train,
...     targets_for_training=balanced_y_train
... )

>>> # Get ensemble predictions.
... ensemble_predictions, ensemble_uncertainty = ue_svm.generate_predictions(
...     inventory_of_estimators=ensembles,
...     features=x_test
... )

>>> # Compute ensemble accuracy.
... ensemble_accuracy = (
... np.sum((y_test.astype(int) == ensemble_predictions).astype(int))
...     / len(ensemble_predictions)
... )
... print(ensemble_accuracy)

.. raw:: latex

    \\newpage

Below we can take a look at the solution produced by the single SVC.

.. figure:: 010_single_svc_predictions.jpg
    :align: center

And here is the confusion matrix for the single SVC.

.. figure:: 010_single_svc_cn_matrix.jpg
    :align: center

.. raw:: latex

    \\newpage

SVC ensemble produces different solution, as we can see from the image
below.

.. figure::  010_ensemble_svm_prediction.jpg
    :align:   center

Also distribution of classes across the prediction is different.

.. figure::  010_ensemble_svm_cn_matrix.jpg
    :align:   center

.. raw:: latex

    \\newpage


Finally we can take a look a te comparison between a single SVC and an
ensemble of SVCs.

.. figure:: 010_comparison.jpg
    :align: center

There is clearly a different fit, which has been achieved on the basis
of training of multiple SVCs on a balanced set, and some of the bias
of the model trained on the imbalanced set has be removed.
However, even though this allows for training of SVMs on larger training
sets, there are two relative problems with this solution: (a) minor
but still present loss in accuracy, and (b) longer execution time
of the ensemble.

References
----------
.. [1] Murphy, K. (2012). Machine Learning A Probabilistic Perspective,
   p. 497
.. [2] Shinmoto Torres, R. L., Ranasinghe, D. C., Shi, Q., Sample, A. P.
   (2013, April). Sensor enabled wearable RFID technology for mitigating
   the risk of falls near beds. In 2013 IEEE International Conference on
   RFID (pp. 191-198). IEEE.
.. [3] Osband, I., Blundell, C., Pritzel, A. Van Roy1, B. (2016). Deep
   Exploration via Bootstrapped DQN.

"""

import io
import os
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
from sklearn.svm import SVC


def get_all_files_within_folder(path, negative_condition):
    """
    Catalogue all files within a folder according to negative condition

    Parameters
    ----------
    path : str
        Path to the folder.
    negative_condition : str
        Exact text contained with the name of the files, which is
        utilized to identify files which will not be catalogued.

    Returns
    -------
    list
        Names of all files within folder except file names designated in
        `negative_condition` parameter.

    """
    return [
        file for file in os.listdir(path) if (os.path.isfile(
            os.path.join(path, file)) and (negative_condition not in file)
        )
    ]


def get_all_sub_folders_within_folder(path):
    """
    Catalogue names of all sub-folders within folder

    Parameters
    ----------
    path : str
        Path to the main folder.

    Returns
    -------
    list
        Names of all sub-folder within folder designated in `path`
        parameter.

    """
    return [
        name for name in os.listdir(path) if
        os.path.isdir(os.path.join(path, name))
    ]


def get_data(folder_wh_data):
    """
    Build an inventory of data sets out of individual files within a
    folder.

    Parameters
    ----------
    folder_wh_data : str
        Folder in which data files are residing.

    Returns
    -------
    list
        All the data sets generated from individual data
        files.

    Notes
    -----
    Function omits files which are having `txt` inside their name.

    """
    # Make inventory of sub folder names.
    inventory_of_sub_folders = get_all_sub_folders_within_folder(
        path=folder_wh_data
    )
    # Declare an empty list to store data files.
    inventory_of_data_sets = []
    # Iterate over sub folder, and then iterate over files within sub-folders,
    # get files, assign correct index, and append to the inventory.
    for sub_folder in inventory_of_sub_folders:
        # Make inventory of data file names.
        inventory_of_data_files = get_all_files_within_folder(
            os.path.join(folder_wh_data, sub_folder),
            negative_condition='txt'
        )
        for data_file_name in enumerate(inventory_of_data_files):
            # Get data.
            print('Getting data from file: %s' % data_file_name[0])
            inventory_of_data_sets.extend([
                pd.read_csv(
                    filepath_or_buffer=os.path.join(
                        folder_wh_data,
                        sub_folder,
                        data_file_name[1]
                    ),
                    header=None,
                    index_col=0
                )
            ])
    return inventory_of_data_sets


class TrainingDataSets:
    """
    Class for generating training, validation, and testing data sets.

    Attributes
    ----------
    original_data : pandas.DataFrame
        Original features and targets.
    indices_of_features : list
        Numeric indication of position of features in `original_data`
    indices_of_targets : list
        Numeric indication of position of targets in `original_data`
    train_features : numpy.array
        Unscaled training features.
    train_targets : numpy.array
        Training targets.
    validation_features : numpy.array
        Validation features
    validation_targets : numpy.array
        Validation Targets
    test_features : numpy.array
        Testing features
    test_targets : numpy.array
        Testing targets
    scaled_train_features : numpy.array
        Train features scaled to mean zero and unit variance. Shuffled
        if desired.
    scaled_validation_features : numpy.array
        Validation features scaled to mean zero and unit variance.
    scaled_test_features : numpy.array
        Test features scaled to mean zero and unit variance.

    """
    def __init__(self, features_and_targets_data_set):
        self.original_data = features_and_targets_data_set
        self.indices_of_features = None
        self.indices_of_targets = None
        self.train_features = None
        self.train_targets = None
        self.validation_features = None
        self.validation_targets = None
        self.test_features = None
        self.test_targets = None
        self.features_mean = None
        self.features_standard_deviation = None
        self.scaled_train_features = None
        self.scaled_validation_features = None
        self.scaled_test_features = None

    def shuffle(self):
        """
        Shuffle scaled features and unscaled targets for training

        Notes
        -----
        Only scaled training features and targets are shuffled. Validation,
        and test data sets are not shuffled.

        """
        np.random.seed(seed=0)
        shuffled_order = np.random.permutation(
            len(self.scaled_train_features)
        )
        self.scaled_train_features = self.scaled_train_features[
            shuffled_order, :
        ]
        self.train_targets = self.train_targets[
            shuffled_order
        ]

    def make_training_data(self, train_size, validation_size):
        """
        Partition the data into training, validation, and testing sets.

        Parameters
        ----------
        train_size : int
            Proportion of the training set.
        validation_size: int
            Proportion of the validation set.

        Notes
        -----
        Size of the testing set is determined implicitly.

        """
        validation_end = train_size + validation_size
        self.train_features = self.original_data[
            0:train_size, self.indices_of_features
        ]
        self.train_targets = self.original_data[
            0:train_size, self.indices_of_targets
        ]
        self.validation_features = self.original_data[
            train_size:validation_end, self.indices_of_features
        ]
        self.validation_targets = self.original_data[
            train_size:validation_end, self.indices_of_targets
        ]
        self.test_features = self.original_data[
            validation_end:, self.indices_of_features
        ]
        self.test_targets = self.original_data[
            validation_end:, self.indices_of_targets
        ]

    def compute_mean_and_standard_deviation(self):
        """
        Computation of mean and standard deviation of features
        in the training data set

        Returns
        -------
        None
            No explicit return.

        Notes
        -----
        Values are stored in the 'features_mean' and
        'features_standard_deviation' attribute of the class.

        """
        self.features_mean = np.mean(
            self.train_features, axis=0
        )
        self.features_standard_deviation = np.std(
            self.train_features, axis=0
        )

    def scale_features(self):
        """
        Standardize features in such manner that their mean is centered
        to zero, and unit of measurement is set to variance.

        Returns
        -------
        numpy.array
            Standardized features are placed inside appropriate
            attributes of the class.

        """
        self.scaled_train_features = (
            self.train_features-self.features_mean
        ) / self.features_standard_deviation
        self.scaled_validation_features = (
            self.validation_features-self.features_mean
        ) / self.features_standard_deviation
        self.scaled_test_features = (
            self.test_features-self.features_mean
        ) / self.features_standard_deviation

    def get_scaled_features(self):
        """
        Convenience method to return scaled features.

        Returns
        -------
        scaled_train_features : numpy.array
            Scaled features for training.
        scaled_validation_features : numpy.array
            Scaled features for validation.
        scaled_test_features : numpy.array
            Scaled features for testing.

        """
        return (
            self.scaled_train_features,
            self.scaled_validation_features,
            self.scaled_test_features
        )

    def get_targets(self):
        """
        Convenience method to return targets and features

        Returns
        -------
        train_targets : numpy.array
            Targets for training.
        validation_targets : numpy.array
            Targets for validation.
        test_targets : numpy.array
            Targets for testing.

        """
        return (
            self.train_targets,
            self.validation_targets,
            self.test_targets
        )


def reduce_set_to_equal_distribution_of_classes(features_for_training,
                                                targets_for_training):
    """
    Reduces the training set to the size N = K x size of the
    least frequent class

    Parameters
    ----------
    features_for_training : numpy.array
        Features which will be used for generating reduced sets.
    targets_for_training : numpy.array
        Targets which will be used for generating reduced sets.

    Returns
    -------
    numpy.array
        Two separate numpy.arrays for features and targets.

    """
    # Compute count of all classes.
    count = np.bincount(targets_for_training.astype(int))
    # Get the smallest class.
    size = int(count[np.argmin(count)] / 2)
    # Declare lists to store new samples.
    inventory_of_features = []
    inventory_of_targets = []

    for label in np.unique(targets_for_training.astype(int)):
        subset_features = features_for_training[
            [item == label for item in targets_for_training]
        ]
        subset_features = subset_features[
            np.random.permutation(len(subset_features))
        ]
        inventory_of_features.extend([subset_features[0:size, ]])

        inventory_of_targets.extend([
            np.reshape(np.array([label] * size), newshape=(-1, 1))
        ])
    # Return after v-stacking.
    return np.vstack(inventory_of_features), np.vstack(inventory_of_targets)


def generate_ensemble(number_of_estimators,
                      features_for_training,
                      targets_for_training):
    """
    Generates a collection of estimators

    Each estimator is trained on a sub-set of the training data, and
    appended to the ensemble. Support Vector Classifier has been
    selected as the classifier of choice, but can be replaced with any
    other classifier.

    Parameters
    ----------
    number_of_estimators : int
        How many estimators will be in the ensemble.
    features_for_training : numpy.array
        Features which will be utilized for training of individual
        estimators.
    targets_for_training : numpy.array
        Targets which will be utilized for training of individual
        estimators.

    Returns
    -------
    list
        A collection of SVCs trained on different sections of features
        and targets pairs.

    Notes
    -----
    The function does not shuffle the data. If shuffling is necessary,
    it has to be done before call to the function.

    """
    size_of_validation_set = int(
        len(features_for_training)
        / number_of_estimators
    )
    # Declare start indices.
    start_indices = list(
        range(
            0,
            len(features_for_training),
            size_of_validation_set
        ),
    )
    # Declare end indices.
    end_indices = list(
        range(
            size_of_validation_set,
            len(features_for_training),
            size_of_validation_set
        )
    )
    # Pop last element of start indices list.
    start_indices.pop(-1)
    # Iterate over start-end indices pairs, subset data, and store
    # trained model into an inventory.
    inventory_of_models = []
    for indices in zip(start_indices, end_indices):
        cv_training_features = np.delete(
            features_for_training, range(indices[0], indices[1]), 0
        )
        cv_training_targets = np.delete(
            targets_for_training, range(indices[0], indices[1]), 0
        )
        cv_estimator = SVC(
            C=1,
            kernel='rbf',
            probability=True,
            random_state=0
        )
        inventory_of_models.extend([
            cv_estimator.fit(
                X=cv_training_features,
                y=cv_training_targets
            )
        ])
    return inventory_of_models


def compute_predictive_entropy(probability):
    """
    Estimate epistemic uncertainty via predictive entropy [1]_

    Parameters
    ----------
    probability : numpy.array
        A numpy.array (N x C) with the probabilities obtained from the
        underlying classier (soft voting).

    Returns
    -------
    numpy.array
        Uncertainty estimate for each prediction.

    Notes
    -----
    For the computation of uncertainty value equal to zero are
    replaced with a small constant near zero.

    References
    ----------
    .. [1] Further details about about predictive entropy available at:
        `https://en.wikipedia.org/wiki/Entropy_(information_theory)`

    """
    return -1 * np.sum(np.log(probability) * probability, axis=0)


def generate_predictions(inventory_of_estimators, features):
    """
    Generate predictions from ensemble

    The function applies 'predict_proba' method to a collection
    of estimators, in order to get predictions and compute uncertainty
    estimate via predictive entropy.

    Parameters
    ----------
    inventory_of_estimators : list
        A collection of estimators placed in a list.
    features : numpy.array
        Features on which to perform prediction.

    Returns
    -------
    ensemble_predictions : numpy.array
        Prediction of class membership.
    uncertainty_estimate : numpy.array
        Uncertainty estimate.

    """
    # Generate predictions.
    inventory_of_probabilities = []
    for ensemble in inventory_of_estimators:
        inventory_of_probabilities.extend(
            [ensemble.predict_proba(X=features)]
        )
    # Reshape probabilities.
    table_of_probabilities = np.array(inventory_of_probabilities)
    # Compute mean of probabilities.
    mean_probabilities = np.mean(table_of_probabilities, axis=0)
    # Replace zero probabilities wh. numerical constant for stability.
    adjusted_probabilities = np.where(
        mean_probabilities == 0,
        1e-9,
        mean_probabilities
    )
    # Get predictions.
    outputs = np.argmax(adjusted_probabilities, axis=1)
    # Get uncertainty estimates.
    uncertainty = np.apply_along_axis(
        compute_predictive_entropy,
        axis=1,
        arr=adjusted_probabilities
    )
    return outputs, uncertainty


def scatter_plot_with_groups(coordinates,
                             labels,
                             legend_colors,
                             legend_descriptions,
                             save_plot=False,
                             path=None):
    """
    Produce scatter plot with coloration according to the labels

    Parameters
    ----------
    coordinates : numpy.array
        Coordinates of points.
    labels : numpy.array
        Vector indicating class membership of each point.
    legend_colors : dict
        Colors to be utilized for coloration of points.
    legend_descriptions : dict
        Labels to be utilized for description in plot legend.
    save_plot : bool
        Indication whether to save a plot. Defaults to none.
    path : str
        Path including the file name where to save the plot.

    Returns
    -------
    None
        No explicit return.

    """
    # Set figure size.
    plt.rcParams['font.serif'] = 'Times New Roman'
    # Set font family.
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.figsize'] = [9.69, 6.27]
    figure, axis = plt.subplots()
    for label in np.unique(labels).astype(int):
        mask = label == labels
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            color=legend_colors[label],
            label=legend_descriptions[label]
        )
    axis.legend()
    plt.show()
    if save_plot is True:
        buffered_image = io.BytesIO()
        plt.savefig(buffered_image, format='png')
        f = io.open(path, 'wb', buffering=0)
        f.write(buffered_image.getvalue())
        f.close()


def make_confusion_matrix(reference, output, prediction_labels):
    """
    Compute confusion matrix

    Parameters
    ----------
    reference : numpy.array
        A vector with reference.
    output : numpy.array
        A vector with targets.
    prediction_labels : list
        Descriptions of labels.

    Returns
    -------
    pandas.DataFrame
         Confuision matrix (see Notes).

    Notes
    -----
    Reference group is placed in row. Proportion of each prediction
    within the reference group is computed across columns
    (horizontally).

    """
    # Counts.
    counts = np.round(np.bincount(reference.astype(int)))
    leading_column_labels = [
        label + ' (n=' + str(counts[index]) + ')'
        for index, label in enumerate(prediction_labels)
    ]

    # Confusion matrix.
    confusion_matrix = pd.crosstab(
        index=output,
        columns=reference,
        normalize='index',
    )
    leading_column = np.hstack(
        (['Reference'], leading_column_labels)
    )
    numeric_columns_with_headings = np.vstack((
        prediction_labels,
        (confusion_matrix.values.round(2)*100).astype(int).astype(str)
    ))
    return np.hstack(
        (leading_column.reshape(-1, 1), numeric_columns_with_headings)
    )


def plot_confusion_matrix(content, save_plot=False, path=None):
    """
    Plot confusion matrix

    Confusion matrix will always be tabulated and plotted. Optionally,
    picture of confusion matrix can be saved.

    Parameters
    ----------
    content : numpy.array
        Numpy array with the complete content of the confusion matrix.
    save_plot : bool
        Indication whether to save the plot. Default set to false.
    path : str
        Path including the file name where to save the plot.

    Returns
    -------
    None
        No explicit return. Optionally plot can be saved.

    Notes
    -----
    Content must consist all numeric content, as well as headings of
    all rows and columns.

    """
    # Set sans-serif font.
    plt.rcParams['font.serif'] = 'Times New Roman'
    # Set font family.
    plt.rcParams['font.family'] = 'serif'
    figure, axis = plt.subplots()
    # Drop axis.
    axis.axis('off')
    # Declare a table object.
    table = axis.table(
        cellText=content,
        loc='center',
        rasterized=True
    )
    # Set properties of cells.
    for cell in table._cells:
        table._cells[cell].set_text_props(linespacing=1)
        table._cells[cell].set_height(.066)
        # Set alignment of leading rows and columns.
        if cell[0] == 0:
            table._cells[cell]._loc = 'center'
        if (cell[1] == 0) and (cell[0] != 0):
            table._cells[cell]._loc = 'left'
    # Set font size.
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    # Utilize maximal space.
    figure.tight_layout(rect=(.01, .01, .99, .99))
    plt.show()
    if save_plot is True:
        buffered_image = io.BytesIO()
        plt.savefig(buffered_image, format='png')
        f = io.open(path, 'wb', buffering=0)
        f.write(buffered_image.getvalue())
        f.close()


def plot_solution(coordinates,
                  original_labels,
                  predicted_labels,
                  legend_colors,
                  legend_descriptions,
                  uncertainty,
                  save=False,
                  path=None):
    """
    Plotting of solution of classification task

    Convenience function to plot: (a) original labels, (b) predicted
    labels, and (c) uncertainty estimate of the model.

    Parameters
    ----------
    coordinates : numpy.array
        Coordinates of labels
    original_labels : numpy.array
        Reference one-dimensional encoding of the class membership.
        One-hot encoding is not supported.
    predicted_labels : numpy.array
        Predicted one-dimensional encoding of the class membership.
        One-hot encoding is not supported.
    legend_colors : dict
        Colors to be utilized for coloration of points.
    legend_descriptions : dict
        Labels to be utilized for description in plot legend.
    uncertainty : numpy.array
        Uncertainty of the models estimate of class membership.
    save : bool
        Option to save the plot. Default set to false.
    path : str
        Absolute path to the file in which to save a plot.

    Returns
    -------
    None
        No explicit return. Plot is displayed on the screen, or saved
        into a file.

    """
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.figsize'] = [9.69, 6.27]
    figure = plt.figure()
    grid = grid_spec.GridSpec(1, 3)
    # Plot reference classification.
    axis_1 = figure.add_subplot(grid[0, 0])
    for label in np.unique(original_labels).astype(int):
        mask = label == original_labels
        axis_1.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            color=legend_colors[label],
            label=legend_descriptions[label]
        )
    axis_1.legend()
    axis_1.set_title('Reference')
    # Plot model solution.
    axis_2 = figure.add_subplot(grid[0, 1])
    for label in np.unique(predicted_labels).astype(int):
        mask = label == predicted_labels
        axis_2.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            color=legend_colors[label],
            label=legend_descriptions[label]
        )
    axis_2.set_title('Prediction')
    # Uncertainty estimate.
    axis_3 = figure.add_subplot(grid[0, 2])
    axis_3.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=uncertainty,
        cmap='RdYlGn_r'
    )
    axis_3.set_title('Uncertainty Estimate')
    plt.show()
    if save is True:
        buffered_image = io.BytesIO()
        plt.savefig(buffered_image, format='png')
        f = io.open(path, 'wb', buffering=0)
        f.write(buffered_image.getvalue())
        f.close()


def plot_comparison(coordinates,
                    reference,
                    solutions,
                    description,
                    coloration,
                    save=True,
                    path=None):
    """
    Plot comparison across different classification solutions.

    Parameters
    ----------
    coordinates : numpy.array
        Coordinates of the points.
    reference : numpy.array
        Labels of the reference.
    solutions : list
        List containing solutions of classification problem.
    description : dict
        Description of each label.
    coloration : dict
        Vector with coloration.
    save : bool
        Option to save the plot.
    path : str
        Path where to save the plot.

    Returns
    -------
    None
        No explicit return. Plot is displayed on the screen, or saved
        into a file.

    """
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.figsize'] = [9.69, 6.27]
    figure = plt.figure()
    grid = grid_spec.GridSpec(1, len(solutions)+1)
    # Plot reference classification.
    axis_1 = figure.add_subplot(grid[0, 0])
    for label in np.unique(reference).astype(int):
        mask = label == reference
        axis_1.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            color=coloration[label],
            label=description[label]
        )
    axis_1.legend()
    axis_1.set_title('Reference')
    # Plot model solution.
    for solution in range(1, len(solutions)+1):
        axis_2 = figure.add_subplot(grid[0, solution])
        for label in np.unique(solutions[solution-1]).astype(int):
            mask = label == solutions[solution-1]
            axis_2.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                color=coloration[label],
                label=description[label]
            )
        axis_2.set_title('Prediction')
    plt.show()
    if save is True:
        buffered_image = io.BytesIO()
        plt.savefig(buffered_image, format='png')
        f = io.open(path, 'wb', buffering=0)
        f.write(buffered_image.getvalue())
        f.close()


def plot_individual_classes(coordinates,
                            class_membership,
                            description,
                            coloration_mode,
                            coloration,
                            save=False,
                            path=None):
    """
    Plot class membership in individual plot

    Point within scatter plots indicating class membership with multiple
    classes can often overlap, therefore debilitating correct analysis.
    This function plots all classes independently.

    Parameters
    ----------
    coordinates : numpy.array
        Coordinates of points.
    class_membership : numpy.array
        Indication of class membership.
    description : dict
        Description of each label.
    coloration_mode : str
        Indication of the mode of coloration.
    coloration : numpy.array
        Vector with coloration.
    save : bool
        Option to save the plot.
    path : str
        Absolute path towards the file in which to save the plot.

    Returns
    -------
    None
        No explicit return.

    """
    number_of_classes = np.unique(class_membership)
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.figsize'] = [9.69, 6.27]
    position = list(product(
        range(0, int(len(number_of_classes)/2)),
        range(0, int(len(number_of_classes)/2))
    ))
    figure = plt.figure()
    grid = grid_spec.GridSpec(2, int(len(number_of_classes)/2))
    for label in number_of_classes.astype(int):
        mask = label == class_membership
        if coloration_mode == 'discrete':
            axis = figure.add_subplot(grid[position[label]])
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                c=coloration[label]
            )
        if coloration_mode == 'continuous':
            axis = figure.add_subplot(grid[position[label]])
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                c=coloration[mask],
                cmap='RdYlGn'
            )
        axis.set_xlim(
            np.min(coordinates[:, 0])*1.2, np.max(coordinates[:, 0])*1.2
        )
        axis.set_ylim(
            np.min(coordinates[:, 1])*1.2, np.max(coordinates[:, 1])*1.2
        )
        axis.set_title('Class: '+description[label])
    plt.show()
    if save is True:
        buffered_image = io.BytesIO()
        plt.savefig(buffered_image, format='png')
        f = io.open(path, 'wb', buffering=0)
        f.close()
        f.write(buffered_image.getvalue())
