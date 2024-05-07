# import pandas as pd
# csv_path = r'..\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\\ConditionNames_SNOMED-CT.csv'

# df = pd.read_csv(csv_path)

# def process_dx(dx_str):
#     dx_list = dx_str.split(',')
#     processed_names = []
#     for code in dx_list:
#         code = int(code)
#         name = df[df['Snomed_CT'] == code]['AcronymName'].values
        
#         # If name is found, append it to the processed_names list
#         if len(name) > 0:
#             processed_names.append(name[0])
#         else:
#             processed_names.append(f"NOF")
    
#     # Join the processed names into a single string separated by commas
#     processed_str = ', '.join(processed_names)
    
#     return processed_str



import pandas as pd
csv_path = r'..\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\\ConditionNames_SNOMED-CT.csv'

df = pd.read_csv(csv_path)

def process_dx(dx_str):
    dx_list = dx_str.split(',')
    processed_names = []
    for code in dx_list:
        code = int(code)
        name = df[df['Snomed_CT'] == code]['AcronymName'].values
        
        # If name is found, append it to the processed_names list
        if len(name) > 0:
            processed_names.append(name[0])
        else:
            processed_names.append(f"NOF")
    
    # Join the processed names into a single string separated by commas
    processed_str = ', '.join(processed_names)
    
    return processed_str