from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.externals import joblib

digits = datasets.load_digits()

Xtrain, Xtest, Ytrain, Ytest = train_test_split(digits.data, digits.target, test_size=0.20, random_state=2)

clf = joblib.load("myresult.pkl")

score = clf.score(Xtest, Ytest)

print("准确率是 {0}".format(score))
