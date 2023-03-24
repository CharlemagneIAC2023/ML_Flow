import mlflow
import mlflow.sklearn

params = {"n_estimators": 5, "random_state": 42}    sk_learn_rfr = RandomForestRegressor(**params)

mlflow.log_params(params)

mlflow.sklearn.log_model(        
        sk_model=sk_learn_rfr,
        artifact_path="sklearn-model",       
        registered_model_name="clement_sk-learn-random-forest-reg-model")
