import pandas as pd

df = pd.read_csv("datos-emergencia-escocia3.csv",parse_dates=[0])
df['Week_Ending_Date'] = df['Week_Ending_Date'].dt.strftime('%Y-%m-%d')
dfsize2=df.groupby(["year"]).sum()["Sum attendance emergency"].reset_index()

print(df.dtypes)

print(pd.__version__)