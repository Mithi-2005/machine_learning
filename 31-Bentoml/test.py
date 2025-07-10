import bentoml #type: ignore
print(bentoml.__version__)
iris_runner=bentoml.sklearn.get('iris_model:latest').to_runner()
iris_runner.init_local()
print(iris_runner.predict.run([[5.9,3,5.1,1.8]]))