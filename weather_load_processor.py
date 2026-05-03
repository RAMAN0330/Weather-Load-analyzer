import warnings
from typing import Dict, List, Optional

import pandas as pd

# from model import *
# from model_1 import *
from model_3 import *
from tabulate import tabulate

from loadProcessor import *
from weatherData import *
from weatherloadIntegrator import *

# from feedback import *  

warnings.simplefilter("ignore")

class WeatherLoadPipeline:
    """End-to-end pipeline for weather-load data processing."""
    
    def __init__(
        self,
        weather_path: str,
        load_path: str,
        state: str = "HARYANA",

    ):
        """
        Initialize the full pipeline.
        
        Args:
            weather_path: Path to weather parquet file
            load_path: Path to load parquet file
            state: State to process
            start_date: Start date for load data
            end_date: End date for load data
        """
        self.weather_processor = WeatherDataProcessor(weather_path, state)
        self.load_processor = LoadDataProcessor(
            load_path, 
            state, 
        )
        self.integrator = None
        
    def run(self, exog_cols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Execute the complete pipeline.
        
        Args:
            exog_cols: Exogenous columns to use
            
        Returns:
            Dictionary with processed weather, load, and integrated data
        """
        print("Processing weather data...")
        weather_processed = self.weather_processor.process()

        print("Processing load data...")
        load_series, load_df = self.load_processor.load_and_process()

        print("Integrating weather and load data...")
        self.integrator = WeatherLoadIntegrator(weather_processed, load_series)
        state_exog = self.integrator.process(exog_cols)
        state_exog = state_exog.sort_values("datetime").reset_index(drop=True)
        state_exogs_forecast = state_exog.tail(96).copy()
        state_exog = state_exog.iloc[:-96].copy()
        
        return {
            'weather': weather_processed,
            'load': load_df,
            'load_series': load_series,
            'state_exog': state_exog,
            'state_exogs_forecast':state_exogs_forecast
        }

    def load_future_weather(self, forecast_weather_path: str) -> pd.DataFrame:
        """
        Load T+1 day weather forecast and preprocess to match model feature format.
        """
        print("Loading T+1 weather forecast...")
        forecast_df = pd.read_parquet(forecast_weather_path)
        forecast_df = self.weather_processor.standardize_weather_columns(forecast_df)
        
        # Ensure datetime alignment
        forecast_df['datetime'] = pd.to_datetime(forecast_df['datetime'])
        forecast_df = forecast_df.set_index('datetime').sort_index()

        print(f"Loaded forecast weather: {forecast_df.shape[0]} records")
        return forecast_df



if __name__ == "__main__":
    pipeline = WeatherLoadPipeline(
        weather_path="C:\\Users\\RamanSharma\\OneDrive - GNA-Energy\\Desktop\\data\\meteo.parquet",
        load_path="C:\\Users\\RamanSharma\\OneDrive - GNA-Energy\\Desktop\\data\\state_load.parquet",
        state="HARYANA",
    )
    
    results = pipeline.run(exog_cols=['temperature', 'precipitation', 'is_weeknd'])

    weather = results['weather']
    load_series = results['load_series']
    state_exog = results['state_exog']
    state_exogs_forecast = results["state_exogs_forecast"]

    data = pd.concat([load_series, state_exog], axis=1)
    data.rename(columns={data.columns[0]: 'load'}, inplace=True)

    data['load'] = pd.to_numeric(data['load'], errors='coerce').fillna(0)
    for col in ['temperature','precipitation','is_weeknd']:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)


    print("Cleaned Data After Preprocessing...")
    print(tabulate(data.head(), headers='keys', tablefmt='psql'))


    metrics, result_df, model, corrector = sarimax_load_forecast_with_seq2seq_residual(
        data, test_size=96
    )


