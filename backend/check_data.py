import pandas as pd
import os

data_path = "../final_data.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    # Get last 10 dates
    dates = sorted(df["date"].unique())
    last_10 = dates[-10:]
    
    print("Last 10 dates total_drawal check:")
    for d in last_10:
        day_df = df[df["date"] == d]
        avg_load = day_df["total_drawal"].mean()
        print(f"{d}: avg_load = {avg_load}")
else:
    print("File not found")
