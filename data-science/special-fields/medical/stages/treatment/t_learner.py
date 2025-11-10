class TLearner():
    """
    T-Learner class.

    Attributes:
      treatment_estimator (object): fitted model for treatment outcome
      control_estimator (object): fitted model for control outcome
    """

    def __init__(self, treatment_estimator, control_estimator):
        """
        Initializer for TLearner class.
        """
        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
        # set the treatment estimator
        self.treatment_estimator = treatment_estimator

        # set the control estimator
        self.control_estimator = control_estimator

        ### END CODE HERE ###

    def predict(self, X):
        """
        Return predicted risk reduction for treatment for given data matrix.

        Args:
          X (dataframe): dataframe containing features for each subject

        Returns:
          preds (np.array): predicted risk reduction for each row of X
        """
        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
        # predict the risk of death using the control estimator
        risk_control = control_estimator.predict_proba(X)

        # predict the risk of death using the treatment estimator
        risk_treatment = treatment_estimator.predict_proba(X)

        # the predicted risk reduction is control risk minus the treatment risk
        pred_risk_reduction = risk_control - risk_treatment

        ### END CODE HERE ###

        return pred_risk_reduction