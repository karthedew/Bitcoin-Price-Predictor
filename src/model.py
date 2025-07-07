import time
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

class Model():

    def __init__(
            self,
            x_data: pd.DataFrame,
            y_data: pd.DataFrame,
            x_scaler: StandardScaler,
            y_scaler: StandardScaler,
            nproc: int = 4
    ):
        self.x_data = x_data
        self.y_data = y_data
        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

        self.nproc = nproc

        self.ada_model = None
        self.rf_model  = None
        self.svm_model = None

        self.ada_val = {
            "MAE": None,
            "MSE": None,
            "R2": None
        }
        self.rf_val = {
            "MAE": None,
            "MSE": None,
            "R2": None
        }
        self.svm_val = {
            "MAE": None,
            "MSE": None,
            "R2": None
        }

        self.model = None
    
    def split_train_test_data(self):
        train_ratio = 0.7

        train_size      = int(len(self.y_data) * train_ratio)

        self.X_train, self.y_train = self.x_data.iloc[:train_size], \
                                     self.y_data.iloc[:train_size]['market_price_usd']
        self.X_test, self.y_test   = self.x_data.iloc[train_size:], \
                                     self.y_data.iloc[train_size:]['market_price_usd']


    def train(self):

        print('Training AdaBoost Model')
        start_time = time.time()
        self.ada_model = self._train_adaboost()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training the AdaBoost Model took {elapsed_time:.4f} seconds")

        print('Training Random Forest Model')
        start_time = time.time()
        self.rf_model  = self._train_random_forest()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training the Random Forest Model took {elapsed_time:.4f} seconds")

        print('Training SVM Model')
        start_time = time.time()
        self.svm_model = self._train_support_vector_machine()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training the SVM Model took {elapsed_time:.4f} seconds")
        
        print('Models:')
        print('ADA: ', self.ada_model)
        print('RF: ', self.rf_model)
        print('SVM: ', self.svm_model)
        print('==================================')

        base_models = [
            ('svm', self.svm_model),
            ('AdaBoost', self.ada_model),
            ('RandomForest', self.rf_model)
        ]

        # Let's weight the MSE's.
        inverse_mse_svm = 1 / self.svm_val['MSE']
        inverse_mse_rf  = 1 / self.rf_val['MSE']
        inverse_mse_ada = 1 / self.ada_val['MSE']

        # Calculate the sum of inverse MSE's
        total_inverse_mse = inverse_mse_svm + inverse_mse_rf + inverse_mse_ada

        # Calculate the weight for each model
        weight_svm = inverse_mse_svm / total_inverse_mse
        weight_rf = inverse_mse_rf / total_inverse_mse
        weight_ada = inverse_mse_ada / total_inverse_mse

        # Define weights based on previous calculation
        weights = [weight_svm, weight_rf, weight_ada]

        voting_reg = VotingRegressor(
            estimators=base_models,
            weights=weights
        )

        voting_reg.fit(self.X_train, self.y_train)

        self.model = voting_reg

    def predict(self, data: pd.DataFrame):
        # Predict
        y_pred  = self.model.predict(data)

        # Inverse transform the predictions to the original target scale
        y_pred_vote  = self.y_scaler.inverse_transform(y_pred.reshape(-1, 1))

        return y_pred_vote

    def save(self):
        dump(self.model, "models/bitcoin_predictor.joblib")


    def load(self, model_path: str | Path = None):
        if not model_path:
            self.model = load("models/bitcoin_predictor.joblib")
        else:
            self.model = load(model_path)

    def check_for_na(self):
        print(self.X_train.isna().sum())
        print(self.y_train.isna().sum())
        print(self.X_test.isna().sum())
        print(self.X_test.isna().sum())

    def _train_adaboost(self):
        # ---------------------------------------------
        # | Setup Tuning Features and Hyperparameters |
        # ---------------------------------------------
        kf = KFold(n_splits=3, shuffle=True, random_state=42) 
        param_grid = {
            'n_estimators': [i for i in range(1, 1000, 200)],
            'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            'loss': ['linear', 'square', 'exponential']
        }
        scoring = {
            "MAE": "neg_mean_absolute_error",
            "MSE": "neg_mean_squared_error",
            "R2": "r2"
        }
        tree_max_depths = [1, 2, 5, 10, 15]
        best_results = []

        # ------------------------------
        # | Perform Tuning of AdaBoost |
        # ------------------------------
        for depth in tree_max_depths:
        
            base_estimator = DecisionTreeRegressor(max_depth=depth)
            ada_model = AdaBoostRegressor(estimator=base_estimator, random_state=42)

            grid_search = GridSearchCV(
                ada_model,
                param_grid,
                cv=kf,
                scoring=scoring,
                n_jobs=self.nproc,
                refit='MSE'
            )

            grid_search.fit(self.X_train, self.y_train)

            best_ada = grid_search.best_estimator_
            y_pred = best_ada.predict(self.X_test)

            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            # Store best results
            best_results.append({
                "max_depth": depth,
                "model": grid_search.best_estimator_,
                "best_params": grid_search.best_params_,
                "best_MAE": mae,
                "best_MSE": mse,
                "best_R2": r2
            })

        # Find best model based on MSE
        best_result = min(best_results, key=lambda x: x["best_MSE"])

        self.ada_val = {
            "MAE": best_result['best_MAE'],
            "MSE": best_result['best_MSE'],
            "R2":  best_result['best_R2']
        }

        return best_result['model']


    def _train_random_forest(self):
        # ---------------------------------------------
        # | Setup Tuning Features and Hyperparameters |
        # ---------------------------------------------
        kf = KFold(n_splits=3, shuffle=True, random_state=42) 
        param_grid = {
            "n_estimators": [i for i in range(1, 1000, 100)],
            "max_depth": [1, 3, 5],
            "max_features": [1, 5, len(self.X_train)],
            "max_leaf_nodes": [2, 3, 5]
        }
        scoring = {
            "MAE": "neg_mean_absolute_error",
            "MSE": "neg_mean_squared_error",
            "R2": "r2"
        }

        # ------------------------------
        # | Perform Tuning of AdaBoost |
        # ------------------------------
        grid_search = GridSearchCV(
                RandomForestRegressor(), 
                param_grid, 
                cv=kf, 
                scoring=scoring,
                refit='MSE',
                n_jobs=self.nproc,
        )
        grid_search.fit(self.X_train, self.y_train)
        # Predict on test data
        y_pred = grid_search.best_estimator_.predict(self.X_test)

        rf_mae = mean_absolute_error(self.y_test, y_pred)
        rf_mse = mean_squared_error(self.y_test, y_pred)
        rf_r2  = r2_score(self.y_test, y_pred)

        self.rf_val = {
            "MAE": rf_mae,
            "MSE": rf_mse,
            "R2":  rf_r2
        }

        return grid_search.best_estimator_

    def _train_support_vector_machine(self):
        # ---------------------------------------------
        # | Setup Tuning Features and Hyperparameters |
        # ---------------------------------------------
        kf = KFold(n_splits=3, shuffle=True, random_state=42) 
        inital_C = [0.01, 0.05, 0.1, 0.5, 1.0, 100, 200, 500, 1000]
        param_grid = {
            "C": [0.01, 0.05, 0.1, 0.5, 1.0, 100, 200, 500, 1000],
            "gamma": [i for i in range(1, 50, 10)],
            "epsilon": [0.001, 0.01, 0.1, 0.5, 1]
        }
        scoring = {
            "MAE": "neg_mean_absolute_error",
            "MSE": "neg_mean_squared_error",
            "R2": "r2"
        }
        # kernels = ['rbf', 'poly', 'linear']
        kernels = ['rbf']


        # ------------------------------
        # | Perform Tuning of AdaBoost |
        # ------------------------------
        best_params = []
        for kernel in kernels:
            svr_model = SVR(kernel=kernel)
            grid_search = GridSearchCV(
                    svr_model, 
                    param_grid, 
                    cv=kf, 
                    scoring=scoring,
                    refit='MSE',
                    n_jobs=self.nproc,
            )
            grid_search.fit(self.X_train, self.y_train)
            best_params.append((kernel, grid_search))

        best_mse = -1
        final_best_model = None
        for kernel, grid_search in best_params:
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(self.X_test)

            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            if mse < best_mse or best_mse == -1:
                best_mse = mse
                final_best_model = best_model
                self.svm_val = {
                    "MAE": mae,
                    "MSE": mse,
                    "R2":  r2
                }

        return final_best_model
