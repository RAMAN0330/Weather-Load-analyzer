from datetime import date
from typing import Tuple

import pandas as pd


class LoadDataProcessor:
    """Processes load demand data."""
    
    def __init__(
        self, 
        load_path: str, 
        state: str = "HARYANA",
        start_date: date = date(2025, 6, 20),
        end_date: date = date(2025, 8, 31)
    ):
        """
        Initialize load data processor.
        
        Args:
            load_path: Path to load parquet file
            state: State name to filter
            start_date: Start date for filtering
            end_date: End date for filtering
        """
        self.load_path = load_path
        self.state = state
        self.start_date = start_date
        self.end_date = end_date
        self.df = None
        
    def load_and_process(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Load and process demand data.
        
        Returns:
            Tuple of (load series indexed by datetime, processed dataframe)
        """
        df = pd.read_parquet(self.load_path, engine='pyarrow')
        print(df.columns)
        df = df[df["state"] == self.state]
        df = df[["date", "time", "time_block", "load_mw"]]
        # df = df[
        #     (df["date"] >= self.start_date) & 
        #     (df["date"] <= self.end_date)
        # ]
        df = df.rename(columns={
            "load_mw": "demand",
            "date": "Date",
            "time_block": "block"
        })
        
        df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['time'].astype(str))

        load_ser = df.set_index('Datetime')['demand']
        self.df = df

        print(df.columns)
        
        return load_ser, df