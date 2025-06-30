import pickle
import numpy as np

class DummyModel:
    def predict(self, X):
        return [42]  # Dummy prediction

class DummyScaler:
    def fit_transform(self, X):
        return X  # No scaling, return input as is

dummy_model = DummyModel()
dummy_scaler = DummyScaler()

with open('model.pkl', 'wb') as f:
    pickle.dump(dummy_model, f)

with open('scale.pkl', 'wb') as f:
    pickle.dump(dummy_scaler, f)
