import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled
