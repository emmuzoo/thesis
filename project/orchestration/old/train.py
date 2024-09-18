import os
import pandas as pd
import xgboost as xgb
import mlflow
import joblib, pickle
import click
from sklearn.metrics import accuracy_score, recall_score


remote_server_uri = "http://localhost:5000"
mlflow.set_tracking_uri(remote_server_uri) # changes
mlflow.set_experiment("xgboost-train")   # changes
mlflow.autolog()                               # changes

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


# Función para entrenar el modelo XGBoost
def train_xgboost(X_train, y_train, X_val, y_val, params=None, num_boost_round=100):
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'max_depth': 5,
            'eta': 0.1,
            'eval_metric': 'logloss'
        }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dtrain, 'train'), (dval, 'eval')]

    with mlflow.start_run():
        bst = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=evals, early_stopping_rounds=10)
        mlflow.xgboost.log_model(bst, "model")
        mlflow.log_params(params)
        mlflow.log_metric("logloss", bst.best_score)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        print("Recall: %.2f%%" % (recall * 100.0))

        example_input = pd.DataFrame(X_train.toarray()[0:5])
        mlflow.xgboost.log_model(bst, artifact_path="xgboost_model", input_example=example_input)


    return bst


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str, params=None, num_boost_round=100):
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'max_depth': 5,
            'eta': 0.1,
            'eval_metric': 'logloss'
        }
    
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "val.pkl"))

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dtrain, 'train'), (dval, 'eval')]

    with mlflow.start_run():  # changes
        #rf = RandomForestRegressor(max_depth=10, random_state=0)
        #rf.fit(X_train, y_train)
        #y_pred = rf.predict(X_val)
        #rmse = mean_squared_error(y_val, y_pred, squared=False)
        #mlflow.log_metric("rmse", rmse)

        bst = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=evals, early_stopping_rounds=10)
        mlflow.xgboost.log_model(bst, "model")
        mlflow.log_params(params)
        mlflow.log_metric("logloss", bst.best_score)
        
        # Predicciones en el conjunto de validación
        y_pred_val = bst.predict(dval)
        y_pred_val_labels = (y_pred_val > 0.5).astype(int)
        
        # Calcular métricas
        accuracy = accuracy_score(y_val, y_pred_val_labels)
        recall = recall_score(y_val, y_pred_val_labels)
        
        # Registrar métricas en MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        print("Recall: %.2f%%" % (recall * 100.0))
        

if __name__ == '__main__':
    run_train()
