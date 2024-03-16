import os
import wfdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
count = 0
# Path to the WFDBRecords folder
data_path = r'..\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords\\'

df_final = pd.DataFrame(columns=["ID", "Age", "Sex", "Dx"])#, "p_signal"

def visualize_ecg(record_path):
    global df_final  # Specify that df_final is global

    record = wfdb.rdrecord(record_path)
    print(record.record_name)
    # print(record.p_signal.shape)
    
    # Extract information from comments
    age = record.comments[0][5:]
    sex = record.comments[1][5:]
    dx = record.comments[2][4:]
    
    
    # Create DataFrame with additional information
    df_data = pd.DataFrame({
        "ID": [record.record_name],
        "Age": [age],
        "Sex": [sex],
        "Dx": [dx],
        # "p_signal": [record.p_signal] 
    })
    # df_data['p_signal'] = df_data['p_signal'].apply(lambda x: x.tolist())
    # print(df_data.isnull().sum())
    df_data.to_csv("dataset.csv", index=False, mode='a', header=False)
    
# Loop through the first-level directories
for dir1 in os.listdir(data_path):
    dir1_path = os.path.join(data_path, dir1)
    
    # Loop through the subdirectories
    for dir2 in os.listdir(dir1_path):
        dir2_path = os.path.join(dir1_path, dir2)

        # Loop through the ECG records
        for file_name in os.listdir(dir2_path):
            if file_name.endswith('.mat'):
                file_path = os.path.join(dir2_path, file_name[:-4])
                visualize_ecg(file_path)


df_final = pd.read_csv("dataset.csv")
# # df_final['p_signal'] = df_final['p_signal'].apply(lambda x: np.array(eval(x)))
# # print(df_final["p_signal"][0])
print(df_final.isnull().sum())
with open('status.txt',"w") as f:
    f.write(str(df_final.isnull().sum()))