from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from abc import *
from config import lgb_params

class BaseTreeModel:
    @abstractmethod
    def fit_model(self):
        pass
    @abstractmethod
    def eval_model(self):
        pass
    @abstractmethod
    def get_model(self):
        pass

class CatBoostClassiferModel(BaseTreeModel):
    def __init__(self):
        self.model = CatBoostClassifier(iterations=30000,
                                        learning_rate=0.01,
                                        task_type='GPU',
                                        auto_class_weights="Balanced",
                                        devices='1',
                                        loss_function='Logloss',
                                        eval_metric='AUC',
                                        early_stopping_rounds=100, )

    def fit_model(self, df_x, df_y, df_valid_x, df_valid_y, cat_features=[]):
        self.model.fit(df_x, df_y,
                       cat_features=cat_features,
                       eval_set=(df_valid_x, df_valid_y),
                       verbose=False,
                       plot=True,
                       use_best_model=True)

    def eval_model(self, x_test):
        score = self.model.predict_proba(x_test)
        return score

    def get_model(self):
        return self.model

class LGBMClassifierModel(BaseTreeModel):
    def __init__(self):
        self.model = LGBMClassifier(**lgb_params)

    def fit_model(self, df_x, df_y, df_valid_x, df_valid_y):
        self.model.fit(df_x,
                       df_y,
                       eval_set=(df_valid_x, df_valid_y))

    def eval_model(self, x_test):
        score = self.model.predict_proba(x_test)
        return score

    def get_model(self):
        return self.model
