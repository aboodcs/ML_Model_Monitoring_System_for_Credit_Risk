import os
import pandas as pd
import numpy as np

os.makedirs('data/raw', exist_ok=True)

columns = [
    'ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 
    'default payment next month'
]

# Generate 1000 random rows
np.random.seed(42)
data = np.random.randint(0, 1000, size=(1000, len(columns)))

df = pd.DataFrame(data, columns=columns)

# Fix the target column to be binary classification (0 or 1)
df['default payment next month'] = np.random.randint(0, 2, size=1000)

# Write to CSV
# The ingestion pipeline expects skiprows=1, so we add a dummy row at the top
with open('data/raw/credit_risk.csv', 'w') as f:
    f.write('Dummy Row to be skipped\n')

df.to_csv('data/raw/credit_risk.csv', mode='a', index=False)

print("Created dummy dataset at data/raw/credit_risk.csv")
