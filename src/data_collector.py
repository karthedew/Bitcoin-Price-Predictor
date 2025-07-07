import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class DataCollector():

    def __init__(self):
        self.data_path = os.path.join(Path(os.getcwd()).parent, 'data')

        self.start_date = pd.to_datetime('2017-01-01')
        self.end_date   = pd.to_datetime('2023-09-02')

        self.dxy_data     = None
        self.bitcoin_data = None
        self.m2sl_data    = None
        self.fng_data     = None

        self.df = None

    def save(self):
        self.df.to_parquet('../data/collected_data.parquet')
    
    def load(self, data_file: str | Path | None = None):
        if not data_file:
            self.df = pd.read_parquet('../data/collected_data.parquet')
        else:
            self.df = pd.read_parquet(data_file)

    def collect(self):
        self.dxy_data     = self._collect_dxy_index_data()
        self.bitcoin_data = self._collect_bitcoin_data()
        self.m2sl_data    = self._collect_m2_money_supply_data()
        self.fng_data     = self._collect_fng_data()
        self.df           = self._data_processing()

    def data_analysis(self):
        self._plot_bitcoin_price()
        self._plot_bitcon_log_and_moving_average()
        self._plot_m2_data()
        self._plot_correlation_heatmap()
        self._plot_fear_and_greed_against_bitcoin_price()
        self._data_metrics()

    def _plot_bitcoin_price(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['date'], self.df['market_price_usd'], label="Bitcoin Price (USD)", color="blue")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.title("Bitcoin Price Over Time")
        plt.legend()

        plt.savefig('../plots/bitcoin_price_plot.png', dpi=300)

    def _plot_bitcon_log_and_moving_average(self):
        df_copy = self.df.copy()
        df_copy['log10_market_price'] = np.log10(df_copy['market_price_usd'])
        # Compute 5-day moving average on log-transformed prices
        df_copy['log10_5_day_MA'] = df_copy['log10_market_price'].rolling(window=5).mean()
        df_copy['log10_7_day_MA'] = df_copy['log10_market_price'].rolling(window=7).mean()

        # Plot log-transformed price and its moving average
        plt.figure(figsize=(12, 6))
        plt.plot(df_copy['date'], df_copy['log10_market_price'], label="Log10 Bitcoin Price (USD)", color="blue", alpha=0.6)
        plt.plot(df_copy['date'], df_copy['log10_5_day_MA'], label="5-Day Moving Average (Log10)", color="red", linewidth=2)
        plt.plot(df_copy['date'], df_copy['log10_7_day_MA'], label="7-Day Moving Average (Log10)", color="green", linewidth=2)
        plt.xlabel("Date")
        plt.ylabel("Log10 Price (USD)")
        plt.title("Log10 - Transformed Bitcoin Price with 5-Day Moving Average")
        plt.legend()
        
        plt.savefig('../plots/trailing_average_plot.png', dpi=300)

    def _plot_m2_data(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['date'], self.df['M2SL'], label="M2 Money Supply", color="red")
        plt.xlabel("Date")
        plt.ylabel("M2 Money Supply")
        plt.title("M2 Money Supply Over Time")
        plt.legend()
        
        plt.savefig('../plots/m2sl_plot.png', dpi=300)

    def _plot_correlation_heatmap(self):
        df_numeric = self.df.select_dtypes(include=["number"])
        df_numeric = df_numeric.drop(columns=['market_price_usd'])
        plt.figure(figsize=(20, 16))
        sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        
        plt.savefig('../plots/correlation_matrix_plot.png', dpi=300)

    def _plot_fear_and_greed_against_bitcoin_price(self):
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Fear & Greed Index", color="red")
        ax1.plot(self.df["date"], self.df["fng_value"], color="red", label="Fear & Greed Index")
        ax1.tick_params(axis="y", labelcolor="red")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Bitcoin Price (USD)", color="blue")
        ax2.plot(self.df["date"], self.df["market_price_usd"], color="blue", label="Bitcoin Price")
        ax2.tick_params(axis="y", labelcolor="blue")

        fig.tight_layout()
        plt.title("Fear & Greed Index vs. Bitcoin Price")
        
        plt.savefig('../plots/fear_and_greed_plot.png', dpi=300)

    def _data_metrics(self):
        # Get the describe output
        desc = self.df.describe()

        # Begin the LaTeX table
        latex_str = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{|l|" + "c|" * len(desc.columns) + "}\n"
        latex_str += "\\hline\n"
        latex_str += "Statistic & " + " & ".join(desc.columns) + " \\\\ \\hline\n"

        # Loop through each row of the description to convert it to LaTeX format
        for index, row in desc.iterrows():
            latex_str += index + " & " + " & ".join([f"{value:.2f}" for value in row]) + " \\\\ \\hline\n"

        # End the LaTeX table
        latex_str += "\\end{tabular}\n\\caption{Summary Statistics}\n\\end{table}"

        print(latex_str)

        return latex_str

    # def df_to_latex_table(df):
    #     # Get the describe output
    #     desc = df.describe()

    #     # Begin the LaTeX table
    #     latex_str = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{|l|" + "c|" * len(desc.columns) + "}\n"
    #     latex_str += "\\hline\n"
    #     latex_str += "Statistic & " + " & ".join(desc.columns) + " \\\\ \\hline\n"

    #     # Loop through each row of the description to convert it to LaTeX format
    #     for index, row in desc.iterrows():
    #         latex_str += index + " & " + " & ".join([f"{value:.2f}" for value in row]) + " \\\\ \\hline\n"

    #     # End the LaTeX table
    #     latex_str += "\\end{tabular}\n\\caption{Summary Statistics}\n\\end{table}"

    #     return latex_str

    def _collect_dxy_index_data(self):
        # years = [
        #     '2014', '2015', '2016',
        #     '2017', '2018', '2019',
        #     '2020', '2021', '2022',
        #     '2023',
        # ]
        csv_files = glob.glob(os.path.join(self.data_path, 'INDEX_US_IFUS_DXY*.csv'))
        dxy_df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)
        dxy_df["Date"] = pd.to_datetime(dxy_df["Date"], format="%m/%d/%Y")
        dxy_df['dxy'] = dxy_df['Close']
        dxy_df = dxy_df.drop(columns=['Open', 'High', 'Low', 'Close'])
        dxy_df = dxy_df.sort_values(by='Date', ascending=False)
        dxy_df = dxy_df.reset_index(drop=True)
        self.dxy_df = dxy_df[
            (dxy_df['Date'] >= self.start_date) &
            (dxy_df['Date'] <= self.end_date)
        ]
        return dxy_df
    
    def _collect_bitcoin_data(self):
        bitcoin_data = pd.read_csv(os.path.join(self.data_path, 'look_into_bitcoin_daily_data.csv'))
        bitcoin_data['datetime'] = pd.to_datetime(bitcoin_data['datetime'])
        bitcoin_data = bitcoin_data[
            (bitcoin_data['datetime'] >= self.start_date) &
            (bitcoin_data['datetime'] <= self.end_date)
        ]
        bitcoin_data = bitcoin_data.drop(columns=['fear_greed_value', 'fear_greed_category'])

        return bitcoin_data
    
    def _collect_fng_data(self):
        fng_data = pd.read_csv(os.path.join(self.data_path, 'CryptoGreedFear.csv'))
        fng_data['date'] = pd.to_datetime(fng_data['date'])
        fng_data = fng_data[
            (fng_data['date'] >= self.start_date) &
            (fng_data['date'] <= self.end_date)
        ]
        fng_data = fng_data.drop(columns=['fng_classification'])

        return fng_data
    
    def _collect_m2_money_supply_data(self):
        m2_supply = pd.read_csv(os.path.join(self.data_path, 'M2SL.csv'))
        m2_supply['observation_date'] = pd.to_datetime(m2_supply['observation_date'])
        m2_supply = m2_supply[
            (m2_supply['observation_date'] >= self.start_date) &
            (m2_supply['observation_date'] <= self.end_date)
        ]
        return m2_supply

    def _data_processing(self):
        self.dxy_data.rename(columns={'Date': 'date'}, inplace=True)
        self.m2sl_data.rename(columns={'observation_date': 'date'}, inplace=True)
        self.bitcoin_data.rename(columns={'datetime': 'date'}, inplace=True)

        dfs = [self.dxy_data, self.m2sl_data, self.bitcoin_data, self.fng_data]

        # Step 1: Create a date range for merging
        date_range = pd.DataFrame({'date': pd.date_range(start=self.start_date, end=self.end_date)})

        for df in dfs:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Step 4: Merge datasets
        merged_df = date_range \
            .merge(self.bitcoin_data, on='date', how='left') \
            .merge(self.fng_data, on='date', how='outer') \
            .merge(self.dxy_data, on='date', how='outer') \
            .merge(self.m2sl_data, on='date', how='outer') \
            
        merged_df = merged_df.ffill()
        merged_df = merged_df.dropna()

        df = None
        df = merged_df.copy()
        df = df[
            (df['date'] >= self.start_date) &
            (df['date'] <= self.end_date)
        ]
        df = df.reset_index(drop=True)
        return df
    
    def check_for_na(self):
        print(self.df.isna().sum())



        
