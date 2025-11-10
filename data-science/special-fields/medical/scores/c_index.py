import numpy as np

def cindex(y_true, scores):
    '''

    Input:
    y_true (np.array): a 1-D array of true binary outcomes (values of zero or one)
        0: patient does not get the disease
        1: patient does get the disease
    scores (np.array): a 1-D array of corresponding risk scores output by the model

    Output:
    c_index (float): (concordant pairs + 0.5*ties) / number of permissible pairs
    '''
    n = len(y_true)
    assert len(scores) == n

    concordant = 0
    permissible = 0
    ties = 0

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # use two nested for loops to go through all unique pairs of patients
    for i in range(n):
        for j in range(i + 1, n):  # choose the range of j so that j>i

            # Check if the pair is permissible (the patient outcomes are different)
            if y_true[i] != y_true[j]:
                # Count the pair if it's permissible
                permissible += 1

                # For permissible pairs, check if they are concordant or are ties

                # check for ties in the score
                if scores[i] == scores[j]:
                    # count the tie
                    ties += 1
                    # if it's a tie, we don't need to check patient outcomes, continue to the top of the for loop.
                    continue

                # case 1: patient i doesn't get the disease, patient j does
                if y_true[i] == 0 and y_true[j] == 1:
                    # Check if patient i has a lower risk score than patient j
                    if scores[i] < scores[j]:
                        # count the concordant pair
                        concordant += 1
                    # Otherwise if patient i has a higher risk score, it's not a concordant pair.
                    # Already checked for ties earlier

                # case 2: patient i gets the disease, patient j does not
                if y_true[i] == 1 and y_true[j] == 0:
                    # Check if patient i has a higher risk score than patient j
                    if scores[i] > scores[j]:
                        # count the concordant pair
                        concordant += 1
                    # Otherwise if patient i has a lower risk score, it's not a concordant pair.
                    # We already checked for ties earlier

    # calculate the c-index using the count of permissible pairs, concordant pairs, and tied pairs.
    c_index = (concordant + 0.5 * ties) / permissible
    ### END CODE HERE ###

    return c_index


def c_for_benefit_score(pairs):
    """
    Compute c-statistic-for-benefit given list of
    individuals matched across treatment and control arms.

    Args:
        pairs (list of tuples): each element of the list is a tuple of individuals,
                                the first from the control arm and the second from
                                the treatment arm. Each individual
                                p = (pred_outcome, actual_outcome) is a tuple of
                                their predicted outcome and actual outcome.
    Result:
        cstat (float): c-statistic-for-benefit computed from pairs.
    """

    # mapping pair outcomes to benefit
    obs_benefit_dict = {
        (0, 0): 0,
        (0, 1): -1,
        (1, 0): 1,
        (1, 1): 0,
    }

    obs_benefit = np.array([obs_benefit_dict[(pair[0][1], pair[1][1])] for pair in pairs])

    # compute average predicted benefit for each pair
    pred_benefit = np.mean(np.array([[pair[0][0], pair[1][0]] for pair in pairs]), axis=-1, keepdims=False)

    concordant_count, permissible_count, risk_tie_count = 0, 0, 0

    # iterate over pairs of pairs
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):

            # if the observed benefit is different, increment permissible count
            if obs_benefit[i] != obs_benefit[j]:

                # increment count of permissible pairs
                permissible_count += 1

                # if concordant, increment count

                if ((pred_benefit[i] > pred_benefit[j]) and (obs_benefit[i] > obs_benefit[j])) or \
                        ((pred_benefit[j] > pred_benefit[i]) and (
                                obs_benefit[j] > obs_benefit[i])):  # change to check for concordance

                    concordant_count += 1

                # if risk tie, increment count
                if pred_benefit[i] == pred_benefit[j]:  # change to check for risk ties
                    risk_tie_count += 1

    # compute c-statistic-for-benefit
    cstat = (concordant_count + risk_tie_count * 0.5) / permissible_count


    return cstat


def c_statistic(pred_rr: list, y: list, w: list, random_seed: int=0):
    """
    Return concordance-for-benefit, the proportion of all matched pairs with
    unequal observed benefit, in which the patient pair receiving greater
    treatment benefit was predicted to do so.

    Args:
        pred_rr (array): array of predicted risk reductions
        y (array): array of true outcomes
        w (array): array of true treatments

    Returns:
        cstat (float): calculated c-stat-for-benefit
    """
    assert len(pred_rr) == len(w) == len(y)
    np.random.seed(random_seed)

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # Collect pred_rr, y, and w into tuples for each patient

    pred_rr = np.array(pred_rr, dtype=np.float32)[..., np.newaxis]
    y = np.array(y, dtype=np.int32)[..., np.newaxis]
    w = np.array(w, dtype=np.int32)[..., np.newaxis]

    tuples = np.hstack((pred_rr, y, w))
    # Collect untreated patient tuples, stored as a list

    untreated = tuples[tuples[:, 2] == 0, :-1]

    # Collect treated patient tuples, stored as a list
    treated = tuples[tuples[:, 2] == 1, :-1]

    # randomly subsample to ensure every person is matched

    # if there are more untreated than treated patients,
    # randomly choose a subset of untreated patients, one for each treated patient (length of treated patients).
    len_treated, len_untreated = treated.shape[0], untreated.shape[0]

    if len(treated) < len(untreated):
        untreated = untreated[np.random.randint(0, len_untreated, len_treated)]

    # if there are more treated than untreated patients,
    # randomly choose a subset of treated patients, one for each untreated patient (length of untreated patients).
    if len(untreated) < len(treated):
        treated = treated[np.random.randint(0, len_treated, len_untreated)]

    assert len(untreated) == len(treated)

    # Sort the untreated patients by their predicted risk reduction
    untreated = np.sort(untreated, axis=0)

    # Sort the treated patients by their predicted risk reduction
    treated = np.sort(treated, axis=0)

    # match untreated and treated patients to create pairs together

    pairs = [(tuple(p[0]), tuple(p[1])) for p in np.stack((untreated, treated), axis=1)]

    # calculate the c-for-benefit using these pairs (use the function that you implemented earlier)
    cstat = c_for_benefit_score(pairs)

    ### END CODE HERE ###

    return cstat