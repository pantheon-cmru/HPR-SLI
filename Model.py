import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#from sklearn.metrics import accuracy_score
#Accuracy metrics
import pickle

df = pd.read_csv('finalDataset.csv')

#print(df.tail())

#features
X = df.drop('class', axis=1)
#target value
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

#print(type(X_test))
#print(X_test.iloc[69])

pipelines = {
    'lr': make_pipeline(StandardScaler(), LogisticRegression(solver='newton-cg', max_iter=100)),
    'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train.values, y_train.values)
    fit_models[algo] = model



#print(fit_models['lr'].predict(X_test))

#accuracy scores -- only works on rf,lr
'''for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))'''

#print(fit_models['rc'].predict(X_test))
#print(y_test)

with open('Final_test_gb.pkl', 'wb') as f:
     pickle.dump(fit_models['gb'], f)

print("Model loaded")
