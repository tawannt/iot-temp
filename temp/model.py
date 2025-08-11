import pandas as pd
from joblib import load
from sklearn.linear_model import LinearRegression

# Tải mô hình đã lưu
model = load('model.joblib')


X_new = pd.DataFrame([[28.0, 77.9, 1.0]],
                     columns=['Temperature', 'Humidity', 'Air_Quality'])

# def predict_temperature(X):
#     pass
# def predict_humidity(X):
#     pass
# def predict_air_quality(X):
#     pass
# def predict_button_pressed(X):
    # pass
def predict_feeding_times(X: pd.DataFrame) -> int:
    result = model.predict(X)
    return result.astype(int)

# Use model to predict the number of time that button is Pressed
result = predict_feeding_times(X_new)
print(result)

