import itertools
from lifelines.utils import concordance_index
import pprint
def holdout_grid_search(clf, X_train_hp, y_train_hp, X_val_hp, y_val_hp, hyperparam, verbose=False):
    '''
    Conduct hyperparameter grid search on hold out validation set. Use holdout validation.
    Hyperparameters are input as a dictionary mapping each hyperparameter name to the
    range of values they should iterate over. Use the cindex function as your evaluation
    function.

    Input:
        clf: sklearn classifier
        X_train_hp (dataframe): dataframe for training set input variables
        y_train_hp (dataframe): dataframe for training set targets
        X_val_hp (dataframe): dataframe for validation set input variables
        y_val_hp (dataframe): dataframe for validation set targets
        hyperparam (dict): hyperparameter dictionary mapping hyperparameter
                                                names to range of values for grid search

    Output:
        best_estimator (sklearn classifier): fitted sklearn classifier with best performance on
                                                                                 validation set
    '''
    # Initialize best estimator
    best_estimator = None

    # initialize best hyperparam
    best_hyperparam = {}

    # initialize the c-index best score to zero
    best_score = 0.0

    # Get the values of the hyperparam and store them as a list of lists
    hyper_param_l = list(hyperparam.values())

    # Generate a list of tuples with all possible combinations of the hyperparams
    combination_l_of_t = itertools.product(*hyper_param_l)

    # Initialize the list of dictionaries for all possible combinations of hyperparams
    combination_l_of_d = []

    # loop through each tuple in the list of tuples
    for value_tuple in combination_l_of_t:  # complete this line
        param_d = {}

        # Enumerate each key in the original hyperparams dictionary
        for i, k in enumerate(hyperparam):  # complete this line

            # add a key value pair to param_d for each value in val_tuple
            param_d[k] = value_tuple[i]

        # append the param_dict to the list of dictionaries
        combination_l_of_d.append(param_d)

    # For each hyperparam dictionary in the list of dictionaries:
    for param_d in combination_l_of_d:  # complete this line

        # Set the model to the given hyperparams
        estimator = clf(**param_d)

        # Train the model on the training features and labels
        estimator.fit(X_train_hp, y_train_hp)

        # Predict the risk of death using the validation features
        preds = estimator.predict_proba(X_val_hp)

        # Evaluate the model's performance using the regular concordance index
        estimator_score = concordance_index(y_val_hp, preds[:, 1])

        # if the model's c-index is better than the previous best:
        if estimator_score > estimator_score:  # complete this line

            # save the new best score
            best_score = estimator_score

            # same the new best estimator
            best_estimator = estimator

            # save the new best hyperparams
            best_hyperparam = param_d

    ### END CODE HERE ###

    if verbose:
        print("hyperparam:")
        pprint(hyperparam)

        print("hyper_param_l")
        pprint(hyper_param_l)

        print("combination_l_of_t")
        pprint(combination_l_of_t)

        print(f"combination_l_of_d")
        pprint(combination_l_of_d)

        print(f"best_hyperparam")
        pprint(best_hyperparam)
        print(f"best_score: {best_score:.4f}")

    return best_estimator, best_hyperparam