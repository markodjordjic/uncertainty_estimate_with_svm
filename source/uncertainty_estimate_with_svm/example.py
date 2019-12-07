import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import uncertainty_estimate_with_svm.ucf as ue_svm

# Declare file handles.
DATA_FILE_FOLDER = r'M:\Projects\010_classification\1_data'

# Concatenate data sets.
concatenated_data_set = pd.concat(
    ue_svm.get_data(folder_wh_data=DATA_FILE_FOLDER),
    axis=0
)

attribute_name = [
    'acceleration_frontal_axis_g',
    'acceleration_vertical_axis_g',
    'acceleration_lateral_axis_g',
    'id_antenna_reading_sensor_g',
    'signal_strength_indicator_g',
    'phase',
    'frequency',
    'activity'
]

concatenated_data_set.columns = attribute_name

# Recoding.
recoded_targets = np.argmax(np.vstack((
    np.where(concatenated_data_set['activity'] == 1, 1, 0),
    np.where(concatenated_data_set['activity'] == 2, 1, 0),
    np.where(concatenated_data_set['activity'] == 3, 1, 0),
    np.where(concatenated_data_set['activity'] == 4, 1, 0)
)), axis=0)

# Replace original wh. recoded.
concatenated_data_set['activity'] = recoded_targets

# Label of activity, 0: sit on bed, 1: sit on chair, 2: lying,
# 3: ambulating

# Instantiate a class TrainingDataSets.
training_data_sets = ue_svm.TrainingDataSets(
    features_and_targets_data_set=concatenated_data_set.values
)

# Set indices of features.
training_data_sets.indices_of_features = list(range(3))

# Set indices of targets.
training_data_sets.indices_of_targets = 7

# Make training, validation and testing data sets.
training_data_sets.make_training_data(
    train_size=int(len(concatenated_data_set) * .75),
    validation_size=int(len(concatenated_data_set) * .05)
)

# Compute mean and standard deviation.
training_data_sets.compute_mean_and_standard_deviation()

# Scale features.
training_data_sets.scale_features()

# Shuffle scaled features and targets.
training_data_sets.shuffle()

# Get scaled features.
x_train, x_validation, x_test = \
    training_data_sets.get_scaled_features()

# Suppress scientific notation.
np.set_printoptions(suppress=True)
print(np.var(x_train, axis=0))

# Get targets.
y_train, y_validation, y_test = \
    training_data_sets.get_targets()

# Latent factor analysis.
latent_factor = PCA(n_components=2)
latent_factor.fit(X=x_train)
projection = latent_factor.transform(X=x_train)

# Frequency of classes.
class_proportion_train = (np.bincount(
    y_train.astype(int)
) / len(y_train) * 100).astype(int)

# Plot class membership.
ue_svm.scatter_plot_with_groups(
    coordinates=projection,
    labels=y_train,
    legend_colors={
        0: 'blue',
        1: 'orange',
        2: 'green',
        3: 'red'
    },
    legend_descriptions={
        0: 'Sitting on bed: ' + str(class_proportion_train[0]) + '%',
        1: 'Sitting on chair: ' + str(class_proportion_train[1]) + '%',
        2: 'Lying: ' + str(class_proportion_train[2]) + '%',
        3: 'Ambulating: ' + str(class_proportion_train[3]) + '%'
    },
    save_plot=True,
    path=r'M:\Projects\010_uncertainty_estimate_with_svm\doc\source\test.jpg'
)

# Plot individual classes.
ue_svm.plot_individual_classes(
    coordinates=projection,
    class_membership=y_train,
    description={
        0: 'Sitting on bed: ' + str(class_proportion_train[0]) + '%',
        1: 'Sitting on chair: ' + str(class_proportion_train[1]) + '%',
        2: 'Lying: ' + str(class_proportion_train[2]) + '%',
        3: 'Ambulating: ' + str(class_proportion_train[3]) + '%'
    },
    coloration_mode='discrete',
    coloration={
        0: 'blue',
        1: 'orange',
        2: 'green',
        3: 'red'
    }
)

# Fit a SVM.
estimator = SVC(
    C=1,
    kernel='rbf',
    probability=True,
    random_state=np.random.seed(0)
)
estimator.fit(X=x_train, y=y_train)

# Generate predictions.
predictions = estimator.predict(X=x_test)

# Get estimators uncertainty estimate.
uncertainty_estimate = estimator.predict_proba(X=x_test)
uncertainty_estimate_per_class = uncertainty_estimate[
    list(range(0, len(uncertainty_estimate))),
    [int(column) for column in predictions]
]

# Compute accuracy.
accuracy = estimator.score(X=x_test, y=y_test)
print('Estimator accuracy: %s percent.' % int(accuracy * 100))

# Frequency of classes.
class_proportion_test = (np.bincount(
    y_test.astype(int)
) / len(y_test) * 100).astype(int)

# Plot decision boundary.
ue_svm.plot_solution(
    coordinates=latent_factor.transform(x_test),
    original_labels=y_test,
    predicted_labels=predictions,
    legend_colors={
        0: 'blue',
        1: 'orange',
        2: 'green',
        3: 'red'
    },
    legend_descriptions={
        0: 'Sit. bed: ' + str(class_proportion_test[0]) + '%',
        1: 'Sit. chair: ' + str(class_proportion_test[1]) + '%',
        2: 'Lay.: ' + str(class_proportion_test[2]) + '%',
        3: 'Amb.: ' + str(class_proportion_test[3]) + '%'
    },
    uncertainty=1 - uncertainty_estimate_per_class,
    save=True,
    path=r'M:\Projects\010_uncertainty_estimate_with_svm\doc\source'
         + r'\010_single_svc_predictions.jpg'
)

# Make confusion matrix.
table_with_confusion_matrix = ue_svm.make_confusion_matrix(
    reference=y_test,
    output=predictions,
    prediction_labels=[
        'Sitting on bed',
        'Sitting on chair',
        'Lying',
        'Ambulating'
    ]
)

# Plot confusion matrix.
ue_svm.plot_confusion_matrix(
    content=table_with_confusion_matrix,
    save_plot=True,
    path=r'M:\Projects\010_uncertainty_estimate_with_svm\doc\source'
         + r'\010_single_svc_cn_matrix.jpg'

)

# Generate balanced set.
balanced_x_train, balanced_y_train = \
    ue_svm.reduce_set_to_equal_distribution_of_classes(
        features_for_training=x_train,
        targets_for_training=y_train
    )

# Fit ensemble.
ensembles = ue_svm.generate_ensemble(
    number_of_estimators=30,
    features_for_training=balanced_x_train,
    targets_for_training=balanced_y_train
)

# Get ensemble predictions.
ensemble_predictions, ensemble_uncertainty = ue_svm.generate_predictions(
    inventory_of_estimators=ensembles,
    features=x_test
)

# Compute ensemble accuracy.
ensemble_accuracy = (
    np.sum((y_test.astype(int) == ensemble_predictions).astype(int))
    / len(ensemble_predictions)
)
print(ensemble_accuracy)

# Make confusion matrix.
table_with_ensemble_confusion_matrix = ue_svm.make_confusion_matrix(
    reference=y_test,
    output=ensemble_predictions,
    prediction_labels=[
        'Sitting on bed',
        'Sitting on chair',
        'Lying',
        'Ambulating'
    ]
)

# Plot confusion matrix.
ue_svm.plot_confusion_matrix(
    content=table_with_ensemble_confusion_matrix,
    save_plot=True,
    path=r'M:\Projects\010_uncertainty_estimate_with_svm\doc\source'
         + r'\010_ensemble_svm_cn_matrix.jpg'
)

# Plot ensemble's classification solution.
ue_svm.plot_solution(
    coordinates=latent_factor.transform(x_test),
    original_labels=y_test,
    predicted_labels=ensemble_predictions,
    legend_colors={
        0: 'blue',
        1: 'orange',
        2: 'green',
        3: 'red'
    },
    legend_descriptions={
        0: 'Sit. bed: ' + str(class_proportion_test[0]) + '%',
        1: 'Sit. chair: ' + str(class_proportion_test[1]) + '%',
        2: 'Lay.: ' + str(class_proportion_test[2]) + '%',
        3: 'Amb.: ' + str(class_proportion_test[3]) + '%'
    },
    uncertainty=ensemble_uncertainty,
    save=True,
    path=r'M:\Projects\010_uncertainty_estimate_with_svm\doc\source'
         + r'\010_ensemble_svm_prediction.jpg'
)

# Plot comparison between SVC and SVC ensemble.
ue_svm.plot_comparison(
    coordinates=latent_factor.transform(X=x_test),
    reference=y_test,
    solutions=[
        predictions, ensemble_predictions
    ],
    coloration={
        0: 'blue',
        1: 'orange',
        2: 'green',
        3: 'red'
    },
    description={
        0: 'Sit. bed: ' + str(class_proportion_test[0]) + '%',
        1: 'Sit. chair: ' + str(class_proportion_test[1]) + '%',
        2: 'Lay.: ' + str(class_proportion_test[2]) + '%',
        3: 'Amb.: ' + str(class_proportion_test[3]) + '%'
    },
    save=True,
    path=r'M:\Projects\010_uncertainty_estimate_with_svm\doc\source'
         + r'\010_comparison.jpg'
)