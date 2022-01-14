#!/usr/bin/env python3

import pandas as pd, matplotlib.pyplot as plt, seaborn as sn, ipaddress
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import svm, metrics, pipeline

df = pd.read_csv('combined.csv')

y_base = df.label

df2 = df.drop(columns=['src_ip','dst_ip','timestamp','label','attack_type'])
#print(df2)

x_train, x_test, y_train, y_test = train_test_split(df2, y_base, test_size=0.2)
m = DecisionTreeClassifier(criterion='entropy',random_state=0)
clf = m.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_predict))

titles, models = [], []
skf = StratifiedKFold(n_splits=10)

case = SelectKBest(chi2, k=15)
case.fit(df2, y_base)
#Top 15 Features
topFeatures = df2[df2.columns[case.get_support()]]
print(topFeatures.columns)
#Correlation Heatmap
correlations = topFeatures.corr()
sn.heatmap(correlations, annot=True)
plt.title('Correlation Heat Map')
plt.show()

titles.append("J48 Decision Tree")
models.append(DecisionTreeClassifier(random_state=0))
titles.append("Support Vector Machines")
models.append(svm.SVC(kernel='linear'))

for m,t in zip(models,titles):
    classifier_pipeline = pipeline.Pipeline([('feature',case),('model',m)])
    y_pred = cross_val_predict(classifier_pipeline, df2, y_base, cv=skf)
    #Accuracy Score based on predicted values
    print("Accuracy:",t, metrics.accuracy_score(y_base, y_pred))
    print("-------------------------------------------")
    scores = cross_val_score(classifier_pipeline, df2, y_base, scoring='f1', cv=skf, n_jobs=1)
    #Scores for top features
    print("Scores for",t, scores)
    print("Mean:",t, scores.mean())
    print("Standard Deviation:",t, scores.std())
    print("-------------------------------------------")
    #Plotting Confusion Matrix (ROC Curve code should be commented)
    #conf_matrix = metrics.confusion_matrix(y_base, y_pred)
    #metrics.ConfusionMatrixDisplay(conf_matrix).plot()
    #plt.title(t+' Confusion Matrix')
    #plt.show()
    #ROC Plotting (Confusion Matrix code should be commented)
    fpr, tpr, threshold = metrics.roc_curve(y_base, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=t+', '+str(m))

plt.title('Receiver Operating Characteristics')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()



