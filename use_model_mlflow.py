import mlflow
logged_model = 'runs:/1b9c22f3667848c7b2ac9bdb07d8c2c4/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
print(type(loaded_model))

# Predict on a Pandas DataFrame.
data = {'x': [1.0, 2.0]}
import pandas as pd
loaded_model.predict(pd.DataFrame(data))