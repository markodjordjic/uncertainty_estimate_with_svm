import io
import os
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
from sklearn.decomposition import PCA
from sklearn.svm import SVC


def get_all_files_within_folder(path, negative_condition):
    """
    Catalogue all files within a folder which naming conforms to negative
    condition

    Parameters
    ----------
    path : string
        Path to the folder.
    negative_condition : sting
        Exact textual content in the name of the files, which is
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
    path : string
        Path to the main folder.

    Returns
    -------
    list
        Names of all sub-folder within folder designated in `path` parameter.

    """
    return [
        name for name in os.listdir(path) if
        os.path.isdir(os.path.join(path, name))
    ]


def get_data(folder_wh_data):
    """
    Build an inventory of data sets out of individual files within a
    folder.

    Function omits files which are having `txt` inside their name.

    Parameters
    ----------
    folder_wh_data : string
        Folder in which data files are residing.

    Returns
    -------
    list
        A list with all the data sets generated from individual data
        files.

    """
    # Make inventory of sub folder names.
    inventory_of_sub_folders = get_all_sub_folders_within_folder(
        path=folder_wh_data
    )
    # Declare an empty list to store data files.
    inventory_of_data_sets = []
    # Iterate over sub folder, and then iterate over files within subfolders,
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
    original_data : pd.DataFrame
        Original features and targets.
    indices_of_features : list
        Numeric indication of position of features in `original_data`
    indices_of_targets : list
        Numeric indication of position of targets in `original_data`
    train_features : np.array
        Unscaled training features.
    train_targets : np.array
        Training targets. Shuffled if desired.
    validation_features : np.array
        Validation features
    validation_targets : np.array
        Validation Targets
    test_features : np.array
        Testing features
    test_targets : np.array
        Testing targets
    scaled_train_features : np.array
        Train features scaled to mean zero and unit variance. Shuffled
        if desired.
    scaled_validation_features : Numpy Array
        Validation features scaled to mean zero and unit variance.
    scaled_test_features : Numpy Array
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
        Shuffle the scaled training features and targets.

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
        Make features and targets

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
        in the training data set.

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
        Numpy Array
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
        scaled_train_features : np.array
            Scaled features for training.
        scaled_validation_features : np.array
            Scaled features for validation.
        scaled_test_features : np.array
            Scaled features for testing.

        """
        return (
            self.scaled_train_features,
            self.scaled_validation_features,
            self.scaled_test_features
        )

    def get_targets(self):
        """
        Convenience method to return targets features.

        Returns
        -------
        train_targets : np.array
            Targets for training.
        validation_targets : np.array
            Targets for validation.
        test_targets : np.array
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

    Firstly, the count of least frequent class is computed. Than a pair
    with features and targets is constructed consisting of samples of
    all classes. Therefore, generated pair is balanced in regards to
    distribution of classes.

    Parameters
    ----------
    features_for_training : np.array
        Features which will be used for generating reduced sets.
    targets_for_training : np.array
        Targets which will be used for generating reduced sets.

    Returns
    -------
    np.array  :
        Two separate np.arrays Features and targets reduced to the size of
        equal to the number of samples belonging to the leas frequent class.

    """
    # Compute count of all classes.
    count = np.bincount(targets_for_training.astype(int))
    # Get the smallest class.
    size = int(count[np.argmin(count)] / 2)
    # Declare lists to store new samples.
    inventory_of_features = []
    inventory_of_targets = []

    for label in np.unique(targets_for_training.astype(int)):
        subset_features = \
            features_for_training[
                [item == label for item in targets_for_training]
            ]
        subset_features = subset_features[
            np.random.permutation(len(subset_features))
        ]
        # np.hstack((
        #     subset_features[0:size, ],
        #     np.reshape(np.array([label] * size), newshape=(-1,1))
        # ))
        inventory_of_features.extend([subset_features[0:size, ]])

        inventory_of_targets.extend([
            np.reshape(np.array([label] * size), newshape=(-1, 1))
        ])
    # Return after vstacking.
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
        How much estimators will be in the ensemble.
    features_for_training : np.array
        Features which will be utilized for training of individual
        estimators.
    targets_for_training : np.array
        Targets which will be utilized for training of individual
        estimators.

    Returns
    -------
    List
        A collection of SVCs trained on different sections of features
        and targets pairs.

    Notes
    -----
        The function does not shuffle the data. If shuffling is
        necessary, it has to be done before call to the function.

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
    probability : np.array
        A np.array (N x C) with the probabilities obtained from the
        underlying classier (soft voting).

    Returns
    -------
    np.array
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
    of estimators, in order to get predictions and computes uncertainty
    estimate via predictive entropy. Along side output, an
    uncertainty estimate is returned as well.

    Parameters
    ----------
    inventory_of_estimators : list
        A collection of estimators placed in a list.
    features : numpy.array
        Features on which to perform prediction.

    Returns
    -------
    ensemble_predictions : np.array
        Prediction of class membership.
    uncertainty_estimate : np.array
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
    Plot scatter plot with coloration according to the labels

    Parameters
    ----------
    coordinates : np.array
        Coordinates of points.
    labels : np.array
        Vector indicating class membership of each point.
    legend_colors : dict
        Colors to be utilized for coloration of points.
    legend_descriptions : dict
        Labels to be utilized for description in plot legend.
    save_plot : bool
        Indication whether to save a plot. Defaults to none
    path : basestring
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
    plt.rcParams['figure.figsize'] = [26.6667*.5, 15*.5]
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
    reference : np.array
        A vector with reference.
    output : np.array
        A vector with targets.
    prediction_labels : list
        Descriptions of labels.

    Returns
    -------
    Confusion matrix as the pd.DataFrame.

    Notes
    -----
    Reference group is placed in row. Proportion of each prediction
    within the reference group is computed across columns (horisontaly).

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
    content : np.array
        Numpy array with the complete content of the confusion matrix.
    save_plot : bool
        Indication whether to save the plot. Default set to false.
    path : basestring
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
    coordinates : np.array
        Coordinates of labels
    original_labels : np.array
        Reference one-dimensional encoding of the class membership.
        One-hot encoding is not supported.
    predicted_labels : np.array
        Predicted one-dimensional encoding of the class membership.
        One-hot encoding is not supported.
    legend_colors : dict
        Colors to be utilized for coloration of points.
    legend_descriptions : dict
        Labels to be utilized for description in plot legend.
    uncertainty : np.array
        Uncertainty of the models estimate of class membership.
    save : bool
        Option to save the plot. Default set to false.
    path : basestring
        Absolute path to the file in which to save a plot.

    Returns
    -------
    None
        No explicit return. Plot is displayed on the screen, or saved
        into a file.

    """
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.figsize'] = [26.6667*.5, 15*.5]
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
    coordinates : np.array
        Coordinates of the points.
    reference : np.array
        Labels of the reference.
    solutions : list
        List containing solutions of classification problem.
    description : dict
        Description of each label.
    coloration : dict
        Vector with coloration.

    Returns
    -------
    None
        No explicit return. Plots are displayed.
    """
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.figsize'] = [26.6667*.5, 15*.5]
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
    classes can often overlap, therefore debilitating correct analysis. This
    function plots all classes independently.

    Parameters
    ----------
    coordinates : np.array
        Coordinates of points.
    class_membership : np.array
        Indication of class membership.
    description : dict
        Description of each label.
    coloration_mode : str
        Indication of the mode of coloration.
    coloration : np.array
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
    plt.rcParams['figure.figsize'] = [26.6667*.5, 15*.5]
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


# if __name__ == '__main__':

    # # Declare file handles.
    # DATA_FILE_FOLDER = r'M:\Projects\010_classification\1_data'

    # # Concatenate data sets.
    # concatenated_data_set = pd.concat(
        # get_data(folder_wh_data=DATA_FILE_FOLDER),
        # axis=0
    # )

    # attribute_name = [
        # 'acceleration_frontal_axis_g',
        # 'acceleration_vertical_axis_g',
        # 'acceleration_lateral_axis_g',
        # 'id_antenna_reading_sensor_g',
        # 'signal_strength_indicator_g',
        # 'phase',
        # 'frequency',
        # 'activity'
    # ]

    # concatenated_data_set.columns = attribute_name

    # # Recoding.
    # recoded_targets = np.argmax(np.vstack((
        # np.where(concatenated_data_set['activity'] == 1, 1, 0),
        # np.where(concatenated_data_set['activity'] == 2, 1, 0),
        # np.where(concatenated_data_set['activity'] == 3, 1, 0),
        # np.where(concatenated_data_set['activity'] == 4, 1, 0)
    # )), axis=0)

    # # Replace original wh. recoded.
    # concatenated_data_set['activity'] = recoded_targets

    # # Label of activity, 0: sit on bed, 1: sit on chair, 2: lying,
    # # 3: ambulating

    # # Instantiate a class TrainingDataSets.
    # training_data_sets = TrainingDataSets(
        # features_and_targets_data_set=concatenated_data_set.values
    # )

    # # Set indices of features.
    # training_data_sets.indices_of_features = list(range(3))

    # # Set indices of targets.
    # training_data_sets.indices_of_targets = 7

    # # Make training, validation and testing data sets.
    # training_data_sets.make_training_data(
        # train_size=int(len(concatenated_data_set)*.75),
        # validation_size=int(len(concatenated_data_set)*.05)
    # )

    # # Compute mean and standard deviation.
    # training_data_sets.compute_mean_and_standard_deviation()

    # # Scale features.
    # training_data_sets.scale_features()

    # # Shuffle scaled features and targets.
    # training_data_sets.shuffle()

    # # Get scaled features.
    # x_train, x_validation, x_test = \
        # training_data_sets.get_scaled_features()

    # # Suppress scientific notation.
    # np.set_printoptions(suppress=True)
    # print(np.var(x_train, axis=0))

    # # Get targets.
    # y_train, y_validation, y_test = \
        # training_data_sets.get_targets()

    # # Latent factor analysis.
    # latent_factor = PCA(n_components=2)
    # latent_factor.fit(X=x_train)
    # projection = latent_factor.transform(X=x_train)

    # # Frequency of classes.
    # class_proportion_train = (np.bincount(
        # y_train.astype(int)
    # )/len(y_train)*100).astype(int)

    # # Plot class membership.
    # scatter_plot_with_groups(
        # coordinates=projection,
        # labels=y_train,
        # legend_colors={
            # 0: 'blue',
            # 1: 'orange',
            # 2: 'green',
            # 3: 'red'
        # },
        # legend_descriptions={
            # 0: 'Sitting on bed: '+str(class_proportion_train[0])+'%',
            # 1: 'Sitting on chair: '+str(class_proportion_train[1])+'%',
            # 2: 'Lying: '+str(class_proportion_train[2])+'%',
            # 3: 'Ambulating: '+str(class_proportion_train[3])+'%'
        # }
    # )

    # # Plot individual classes.
    # plot_individual_classes(
        # coordinates=projection,
        # class_membership=y_train,
        # description={
            # 0: 'Sitting on bed: '+str(class_proportion_train[0])+'%',
            # 1: 'Sitting on chair: '+str(class_proportion_train[1])+'%',
            # 2: 'Lying: '+str(class_proportion_train[2])+'%',
            # 3: 'Ambulating: '+str(class_proportion_train[3])+'%'
        # },
        # coloration_mode='discrete',
        # coloration={
            # 0: 'blue',
            # 1: 'orange',
            # 2: 'green',
            # 3: 'red'
        # }
    # )

    # # Fit a SVM.
    # estimator = SVC(
        # C=1,
        # kernel='rbf',
        # probability=True,
        # random_state=np.random.seed(0)
    # )
    # estimator.fit(X=x_train, y=y_train)

    # # Generate predictions.
    # predictions = estimator.predict(X=x_test)

    # # Get estimators uncertainty estimate.
    # uncertainty_estimate = estimator.predict_proba(X=x_test)
    # uncertainty_estimate_per_class = uncertainty_estimate[
        # list(range(0, len(uncertainty_estimate))),
        # [int(column) for column in predictions]
    # ]

    # # Compute accuracy.
    # accuracy = estimator.score(X=x_test, y=y_test)
    # print('Estimator accuracy: %s percent.' % int(accuracy*100))

    # # Frequency of classes.
    # class_proportion_test = (np.bincount(
        # y_test.astype(int)
    # ) / len(y_test) * 100).astype(int)

    # # Plot decision boundary.
    # plot_solution(
        # coordinates=latent_factor.transform(x_test),
        # original_labels=y_test,
        # predicted_labels=predictions,
        # legend_colors={
            # 0: 'blue',
            # 1: 'orange',
            # 2: 'green',
            # 3: 'red'
        # },
        # legend_descriptions={
            # 0: 'Sit. bed: '+str(class_proportion_test[0])+'%',
            # 1: 'Sit. chair: '+str(class_proportion_test[1])+'%',
            # 2: 'Lay.: '+str(class_proportion_test[2])+'%',
            # 3: 'Amb.: '+str(class_proportion_test[3])+'%'
        # },
        # uncertainty=1-uncertainty_estimate_per_class
    # )

    # # Make confusion matrix.
    # table_with_confusion_matrix = make_confusion_matrix(
        # reference=y_test,
        # output=predictions,
        # prediction_labels=[
            # 'Sitting on bed',
            # 'Sitting on chair',
            # 'Lying',
            # 'Ambulating'
        # ]
    # )

    # # Plot confusion matrix.
    # plot_confusion_matrix(content=table_with_confusion_matrix)

    # # Generate balanced set.
    # balanced_x_train, balanced_y_train = \
        # reduce_set_to_equal_distribution_of_classes(
            # features_for_training=x_train,
            # targets_for_training=y_train
        # )

    # # Fit ensemble.
    # ensembles = generate_ensemble(
        # number_of_estimators=30,
        # features_for_training=balanced_x_train,
        # targets_for_training=balanced_y_train
    # )

    # # Get ensemble predictions.
    # ensemble_predictions, ensemble_uncertainty = generate_predictions(
        # inventory_of_estimators=ensembles,
        # features=x_test
    # )

    # # Compute ensemble accuracy.
    # ensemble_accuracy = (
        # np.sum((y_test.astype(int) == ensemble_predictions).astype(int))
        # / len(ensemble_predictions)
    # )
    # print(ensemble_accuracy)

    # # Make confusion matrix.
    # table_with_ensemble_confusion_matrix = make_confusion_matrix(
        # reference=y_test,
        # output=ensemble_predictions,
        # prediction_labels=[
            # 'Sitting on bed',
            # 'Sitting on chair',
            # 'Lying',
            # 'Ambulating'
        # ]
    # )

    # # Plot confusion matrix.
    # plot_confusion_matrix(
        # content=table_with_ensemble_confusion_matrix
    # )

    # # Plot ensemble's classification solution.
    # plot_solution(
        # coordinates=latent_factor.transform(x_test),
        # original_labels=y_test,
        # predicted_labels=ensemble_predictions,
        # legend_colors={
            # 0: 'blue',
            # 1: 'orange',
            # 2: 'green',
            # 3: 'red'
        # },
        # legend_descriptions={
            # 0: 'Sit. bed: '+str(class_proportion_test[0])+'%',
            # 1: 'Sit. chair: '+str(class_proportion_test[1])+'%',
            # 2: 'Lay.: '+str(class_proportion_test[2])+'%',
            # 3: 'Amb.: '+str(class_proportion_test[3])+'%'
        # },
        # uncertainty=ensemble_uncertainty
    # )

    # # Plot comparison between SVC and SVC ensemble.
    # plot_comparison(
        # coordinates=latent_factor.transform(X=x_test),
        # reference=y_test,
        # solutions=[
            # predictions, ensemble_predictions
        # ],
        # coloration={
            # 0: 'blue',
            # 1: 'orange',
            # 2: 'green',
            # 3: 'red'
        # },
        # description={
            # 0: 'Sit. bed: '+str(class_proportion_test[0])+'%',
            # 1: 'Sit. chair: '+str(class_proportion_test[1])+'%',
            # 2: 'Lay.: '+str(class_proportion_test[2])+'%',
            # 3: 'Amb.: '+str(class_proportion_test[3])+'%'
        # }
    # )
