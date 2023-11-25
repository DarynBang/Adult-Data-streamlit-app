import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

hide_footer_style = """
    <style>
    footer {visibility: hidden;}
    </style>
"""

st.set_page_config(layout='wide')
st.markdown(hide_footer_style, unsafe_allow_html=True)

X, y = st.session_state[['X', 'y']]

st.header("Predictive Modeling/Machine Learning")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
def Scaling_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


accuracies= []
recalls = []
precisions = []
models = []

from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
def model_evaluation(model, predictions, actual_values):
    models.append(model)
    accuracies.append(accuracy_score(predictions, actual_values))
    recalls.append(recall_score(predictions, actual_values))
    precisions.append(precision_score(predictions, actual_values))


X_train_te, X_test_te, y_train_te, y_test_te = Scaling_data(X, y)

from sklearn.svm import SVC

svc_clf = SVC(random_state=19)
svc_clf.fit(X_train_te, y_train_te)
svc_predictions = svc_clf.predict(X_test_te)
model_evaluation("SVC", svc_predictions, y_test_te)


from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression()
log_clf.fit(X_train_te, y_train_te)
log_predictions = log_clf.predict(X_test_te)
model_evaluation("Logistic Regression", log_predictions, y_test_te)


from sklearn.ensemble import GradientBoostingClassifier
gdb_clf = GradientBoostingClassifier(random_state=19)
gdb_clf.fit(X_train_te, y_train_te)
gdb_predictions = gdb_clf.predict(X_test_te)
model_evaluation("GDB", gdb_predictions, y_test_te)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=19)

from sklearn.model_selection import RandomizedSearchCV

parameters = [
    {'n_estimators': [3, 10, 20, 30, 50, 80], 'max_features': [2, 4, 6, 8], 'max_depth': [4, 5]},
    {'bootstrap': [False], 'n_estimators': [3, 5, 10], 'max_features': [2, 3, 4], 'max_depth': [4, 5]}
]
randomized = RandomizedSearchCV(estimator=rfc, param_distributions=parameters, n_jobs=-1, cv=5,
                          return_train_score=True, scoring='accuracy', random_state=20)
randomized.fit(X_train_te, y_train_te)
forest_clf = randomized.best_estimator_
forest_predictions = forest_clf.predict(X_test_te)
model_evaluation("RFC", forest_predictions, y_test_te)


# fig, axes = plt.subplots(2, 2)
#
# sns.heatmap(data=confusion_matrix(svc_predictions, y_test_te), annot=True, cmap='coolwarm', ax=axes[0, 0], fmt='g')
# axes[0, 0].set_title('SVC')
#
# sns.heatmap(data=confusion_matrix(log_predictions, y_test_te), annot=True, cmap='coolwarm', ax=axes[0, 1], fmt='g')
# axes[0, 1].set_title('Logistic')
#
# sns.heatmap(data=confusion_matrix(gdb_predictions, y_test_te), annot=True, cmap='coolwarm', ax=axes[1, 0], fmt='g')
# axes[1, 0].set_title('GDBoost')
#
# sns.heatmap(data=confusion_matrix(rfc_predictions, y_test_te), annot=True, cmap='coolwarm', ax=axes[1, 1], fmt='g')
# axes[1, 1].set_title('RFC')
#
# plt.show()

predictions = [svc_predictions, log_predictions, gdb_predictions, forest_predictions]
model_names = ['SVC', 'Logistic', 'GDBoost', 'RFC']

st.subheader("Confusion Matrix")
# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Iterate over models and plot confusion matrix
for ax, predictions, name in zip(axes.flatten(), predictions, model_names):
    cm = confusion_matrix(y_test_te, predictions)
    sns.heatmap(data=cm, annot=True, cmap='viridis', ax=ax, fmt='g')
    ax.set_title(name)

plt.tight_layout()
st.pyplot(fig)

st.markdown("----")
st.markdown("Metric results")
results = pd.DataFrame(data={'Accuracy': accuracies, 'Precision': precisions, 'Recall': recalls}, index=models)
styled_results = results.style.highlight_max(axis=0, color='purple', subset=pd.IndexSlice[:, :])

st.write(styled_results)

