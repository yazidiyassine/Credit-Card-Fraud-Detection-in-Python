# Fit and Predict
rfc = RandomForestClassifier()
rfc.fit(dp.X_train, dp.Y_train)
y_pred = rfc.predict(dp.x_test)


# For the performance let's use some metrics from SKLEARN module
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("The accuracy score is: ", accuracy_score(dp.y_test, y_pred))
print("The precision score is: ", precision_score(dp.y_test, y_pred))
print("The recall score is: ", recall_score(dp.y_test, y_pred))
print("The f1 score is: ", f1_score(dp.y_test, y_pred))
