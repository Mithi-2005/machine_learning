import bentoml # type: ignore
from sklearn.svm import SVC
from sklearn.datasets import load_iris
iris=load_iris()

x,y=iris.data,iris.target

clf=SVC(gamma='scale')
clf.fit(x,y)

saved_model=bentoml.sklearn.save_model("iris_model",clf)
print('Model saved as ',saved_model)


### iris_model:lpw3a5c5ls4zx6h6
