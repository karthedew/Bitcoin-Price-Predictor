import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class DataPreprocessing():

    def __init__(self, data: pd.DataFrame):
        self.df = data
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.x_scaled_data = None
        self.y_scaled_data = None

    def save(self):
        self.df.to_parquet('../data/processed_data.parquet')
    
    def load(self, data_file: str | Path):
        self.df = pd.read_parquet('../data/processed_data.parquet')
    
    def process(self):
        df_normalized = self.df.copy()
        df_normalized['previous_price'] = df_normalized['market_price_usd'].shift(1)
        df_normalized['5_day_avg'] = df_normalized['market_price_usd'].rolling(window=5).mean()
        df_normalized.bfill(inplace=True)
        reference_date = df_normalized['date'].min()
        df_normalized['days_since'] = (df_normalized['date'] - reference_date).dt.days
        df_normalized = df_normalized.drop(columns=['date'])

        y_df = pd.DataFrame(df_normalized['market_price_usd'], columns=['market_price_usd'])
        x_df = df_normalized.drop(columns=['market_price_usd'])

        df_x_scaled = pd.DataFrame(self.x_scaler.fit_transform(x_df), columns=x_df.columns)
        df_y_scaled = pd.DataFrame(self.y_scaler.fit_transform(y_df), columns=y_df.columns)

        self.x_scaled_data = df_x_scaled
        self.y_scaled_data = df_y_scaled

    def validate(self):
        return
    
    def check_for_na(self):
        print('x_scaled_data: ', self.x_scaled_data.isna().sum())
        print('y_scaled_data: ', self.y_scaled_data.isna().sum())
