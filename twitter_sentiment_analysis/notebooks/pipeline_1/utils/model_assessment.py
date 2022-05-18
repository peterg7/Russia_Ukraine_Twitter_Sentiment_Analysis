
# Import dependencies
from shared_imports import np
from sklearn.metrics import confusion_matrix, classification_report


def performanceSummary(model, X_test, y_test):
    # Predict
    y_pred = model.predict(X_test)

    # Build confusion matrix to evaluate the model results
    confusion = confusion_matrix(y_test, y_pred, labels=np.unique(y_pred))

    # Get classification report
    classification = classification_report(y_test, y_pred, labels=np.unique(y_pred))

    # Use score method to get accuracy of model
    acc_score = model.score(X_test, y_test)

    return {
        'confusion': confusion,
        'classification': classification,
        'acc_score': acc_score,
    }