from sdv.timeseries import PAR
import pandas as pd
import numpy as np
import joblib
import time

def get_monthly_avg(data):
    data["month"] = data["timeStamp"].dt.month
    data = data[["month", "temp"]].groupby("month")
    data = data.agg({"temp": "mean"})
    return data

def multiTSDataStream():
    multi_ts_gen = PAR.load('Multivar_TS_demo/new_data_generator.sav')
    generate_df = multi_ts_gen.sample(1)   
    monthly_avg = get_monthly_avg(generate_df).to_dict().get("temp")
    
    def above_monthly_avg(date, temp):
        month = date.month
        return 1 if temp > monthly_avg.get(month) else 0
    
    generate_df["temp_above_monthly_avg"] = generate_df.apply(
        lambda x: above_monthly_avg(x["timeStamp"], x["temp"]), axis=1
    )
    
    del generate_df["month"]
    generate_df = generate_df.reset_index().iloc[:, 1:]
    return generate_df



