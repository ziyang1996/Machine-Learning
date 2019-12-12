# scikit-learn k-fold cross-validation
from numpy import array
from sklearn.model_selection import KFold
# data sample
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# prepare cross validation
kfold = KFold(n_splits=3, shuffle = True, random_state= 1)
# enumerate splits
for train, test in kfold.split(data):
    print('train: %s, test: %s' % (data[train], data[test]))



from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
X, y = iris.data, iris.target

clf = svm.SVC(kernel='linear', C=1, random_state=0)

n_folds = 5
kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X)
scores = cross_val_score(clf, X, y, scoring='precision_macro', cv = kf)

print(scores[0])