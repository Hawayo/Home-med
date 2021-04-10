from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB
from sklearn import metrics
import numpy as np

training_data = ''
testing_data = ''
with open('trainBayesian.csv') as trainingFile:
    training_data = np.loadtxt(trainingFile, delimiter=",")

data = training_data[:,1:89]
target = training_data[:,90]

#test_size 0.3
print("################## test_size = 0.3 #################")
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3,random_state=1)

gnb = GaussianNB()
y_predG = gnb.fit(X_train, y_train).predict(X_test)
print("Gaussian Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_predG).sum()))
print("Gaussian Accuracy:",metrics.accuracy_score(y_test, y_predG))
y_predG = gnb.fit(X_train, y_train).predict_proba(X_test)
np.savetxt("proba30.txt",y_predG)

cnb = CategoricalNB()
y_predC = cnb.fit(X_train, y_train).predict(X_test)
print("Categorical Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_predC).sum()))
print("Categorical Accuracy:",metrics.accuracy_score(y_test, y_predC))

mnb = MultinomialNB()
y_predM = mnb.fit(X_train, y_train).predict(X_test)
print("Multinomial Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_predM).sum()))
print("Categorical Accuracy:",metrics.accuracy_score(y_test, y_predM))

#test_size 0.5
print("################## test_size = 0.5 #################")
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.5,random_state=1)

gnb = GaussianNB()
y_predG = gnb.fit(X_train, y_train).predict(X_test)
print("Gaussian Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_predG).sum()))
print("Gaussian Accuracy:",metrics.accuracy_score(y_test, y_predG))
y_predG = gnb.fit(X_train, y_train).predict_proba(X_test)
np.savetxt("proba50.txt",y_predG)

cnb = CategoricalNB()
y_predC = cnb.fit(X_train, y_train).predict(X_test)
print("Categorical Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_predC).sum()))
print("Categorical Accuracy:",metrics.accuracy_score(y_test, y_predC))

mnb = MultinomialNB()
y_predM = mnb.fit(X_train, y_train).predict(X_test)
print("Multinomial Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_predM).sum()))
print("Categorical Accuracy:",metrics.accuracy_score(y_test, y_predM))

#test_size 0.97. Gaussian is more accurate than Categorical & Multinomial
print("################## test_size = 0.97 #################")
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.97,random_state=1)

gnb = GaussianNB()
y_predG = gnb.fit(X_train, y_train).predict(X_test)
print("Gaussian Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_predG).sum()))
print("Gaussian Accuracy:",metrics.accuracy_score(y_test, y_predG))
y_predG = gnb.fit(X_train, y_train).predict_proba(X_test)
np.savetxt("proba97.txt",y_predG)

cnb = CategoricalNB()
y_predC = cnb.fit(X_train, y_train).predict(X_test)
print("Categorical Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_predC).sum()))
print("Categorical Accuracy:",metrics.accuracy_score(y_test, y_predC))

mnb = MultinomialNB()
y_predM = mnb.fit(X_train, y_train).predict(X_test)
print("Multinomial Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_predM).sum()))
print("Categorical Accuracy:",metrics.accuracy_score(y_test, y_predM))

#test_size 0.98. Gaussian NB produced <1.0 accuracy
print("################## test_size = 0.98 #################")
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.98,random_state=1)

gnb = GaussianNB()
y_predG = gnb.fit(X_train, y_train).predict(X_test)
print("Gaussian Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_predG).sum()))
print("Gaussian Accuracy:",metrics.accuracy_score(y_test, y_predG))
y_predG = gnb.fit(X_train, y_train).predict_proba(X_test)
np.savetxt("proba98.txt",y_predG)

mnb = MultinomialNB()
y_predM = mnb.fit(X_train, y_train).predict(X_test)
print("Multinomial Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_predM).sum()))
print("Multinomial Accuracy:",metrics.accuracy_score(y_test, y_predM))


