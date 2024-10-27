import pandas as pd
from matplotlib import pyplot as plt
logger_fname = "log_compute.csv"
logger_df = pd.read_csv(logger_fname)

print("Avg CPU usage", logger_df['CPU (%)'].mean())
print("Avg RAM usage", logger_df['RAM (%)'].mean())


t = pd.to_datetime(logger_df['Timestamp (s)'], unit='s')
cols = [col for col in logger_df.columns
        if 'time' not in col.lower() and 'temp' not in col.lower()]
plt.figure(figsize=(15, 9))
plt.plot(t, logger_df[cols])
plt.legend(cols)
plt.xlabel('Time')
plt.ylabel('Utilisation (%)')
plt.show()


for col in logger_df.columns:
    if 'time' in col.lower(): continue
    plt.figure(figsize=(15, 9))
    plt.plot(t, logger_df[col])
    plt.xlabel('Time')
    plt.ylabel(col)
    plt.show()