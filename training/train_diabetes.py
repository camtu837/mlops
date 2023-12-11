# train_diabetes.py

from azureml.core import Workspace, Dataset, Run
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from joblib import dump
import os

run = Run.get_context()
exp = run.experiment
ws = run.experiment.workspace

print("Loading training data...")
datastore = ws.get_default_datastore()
datastore_paths = [(datastore, 'diabetes/diabetes.csv')]
traindata = Dataset.Tabular.from_delimited_files(path=datastore_paths)
diabetes = traindata.to_pandas_dataframe()
print("Columns:", diabetes.columns) 
print("Diabetes data set dimensions: {}".format(diabetes.shape))

y = diabetes.pop('Y')
X_train, X_test, y_train, y_test = train_test_split(diabetes, y, test_size=0.2, random_state=0)
data = {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test, "y": y_test}}

print("Training the model...")
alpha = 0.1  # For simplicity, you can set your desired alpha value
print("alpha:", alpha)
run.log("alpha", alpha)
reg = Ridge(alpha=alpha)
reg.fit(data["train"]["X"], data["train"]["y"])
run.log_list("coefficients", reg.coef_)

print("Evaluate the model...")
preds = reg.predict(data["test"]["X"])
mse = mean_squared_error(preds, data["test"]["y"])
print("Mean Squared Error:", mse)
run.log("mse", mse)

# Save model as part of the run history
print("Exporting the model as a pickle file...")
outputs_folder = './model'
os.makedirs(outputs_folder, exist_ok=True)

model_filename = "sklearn_diabetes_model.pkl"
model_path = os.path.join(outputs_folder, model_filename)
dump(reg, model_path)

# Upload the model file explicitly into artifacts
print("Uploading the model into run artifacts...")
run.upload_file(name="./outputs/models/" + model_filename, path_or_stream=model_path)
print("Uploaded the model {} to experiment {}".format(model_filename, run.experiment.name))

run.complete()
