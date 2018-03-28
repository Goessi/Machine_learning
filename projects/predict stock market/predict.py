import pandas as pd
from datetime import datetime
from datetime import datetime
import numpy as np
df = pd.read_csv('sphist.csv')
df['Date'] = pd.to_datetime(df['Date'])
a = df['Date']>datetime(year=2015,month=4,day=1)
df = df.sort_values(by=['Date'],ascending=True)

# I found if you use double for loop, it always go slow to the end of the world :)

df['day_5'] = data['Close'].rolling(5).mean()
df['day_5'].shift()