import pandas as pd 
import numpy as np

#st.subheader("Attendance in emergencies per year")
df = pd.read_csv("Hospital_escocia_emergencias_3.csv",parse_dates=[0])
numeric_columns = df.select_dtypes(include=np.number).columns.tolist() # added
dfsize2 = df.groupby(["year"], as_index=False)["Sum attendance emergency"].sum() # modified
print(dfsize2)
# dfsize2=df.groupby(["year"]).sum()["Sum attendance emergency"].reset_index()
#fig1 = px.bar(dfsize2, y='Sum attendance emergency',color="Sum attendance emergency",x="year",height=400)
#st.plotly_chart(fig1,use_container_width=True)