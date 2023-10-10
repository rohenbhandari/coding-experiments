# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:09:16 2023

@author: rb
"""

import pandas as pd

# Define synthetic data
data = {
    'email': [
        'Cheap luxury watches 50% off',
        'Update your account information',
        'Your invoice is attached',
        'Win a free iPhone now',
        'Meeting rescheduled for tomorrow',
        'Project deadline extended',
        'Get a loan with 0% interest',
        'Your bank statement is attached',
        'Congratulations, you won a prize',
        'Important update regarding your account'
    ],
    'label': [1, 1, 0, 1, 0, 0, 1, 0, 1, 0]  # 1 for spam, 0 for not spam
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('emails.csv', index=False)