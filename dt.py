import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import graphviz
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

train = pd.read_csv('trainDT.csv')

y = train.prognosis
x = train.iloc[:, :89]

#test_size 0.3
print("################## test_size = 0.3 #################")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
clf = DecisionTreeClassifier(criterion="entropy", max_depth=16, random_state=1)
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print("Accuracy @ test_size=0.3:",metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = train.columns[:89], class_names = y)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree30.png')
Image(graph.create_png())

#test_size 0.5
print("################## test_size = 0.5 #################")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)
clf = DecisionTreeClassifier(criterion="entropy", max_depth=16, random_state=1)
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print("Accuracy @ test_size=0.5:",metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = train.columns[:89], class_names = y)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree50.png')
Image(graph.create_png())