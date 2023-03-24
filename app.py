import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston


boston = load_boston()
X_train = boston.data
y_train = boston.target

params = {"n_estimators": 5, "random_state": 42}

mlflow.start_run()
mlflow.log_params(params)

sk_learn_rfr = RandomForestRegressor(**params)
sk_learn_rfr.fit(X_train, y_train)

mlflow.sklearn.log_model(
    sk_model=sk_learn_rfr,
    artifact_path="sklearn-model",
    registered_model_name="clement_sk-learn-random-forest-reg-model"
)

mlflow.end_run()

