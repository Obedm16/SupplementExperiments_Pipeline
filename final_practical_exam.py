# function merge_all_data transforms, maniplate and returns desired defined dataframe

import pandas as pd
import numpy as np


# Let's explore data type of each file
print("\nExloring Data Types For The Four Datasets")
print("\nuser_health:")
user_health = pd.read_csv('user_health_data.csv')
print(user_health.dtypes)
print("\nuser_health dataframe:")
print(user_health)
print('*****'*5)
print('\nDate Colunm in User Health')
print(user_health['date'])
print('\nsleep_hours Colunm in User Health')
print(user_health['sleep_hours'])

print("\nsupplement:")
supplement = pd.read_csv('supplement_usage.csv')
print(supplement.dtypes)
print("\nsupplement dataframe:")
print(supplement)
print('*****'*5)
print('\nDate Colunm in supplement')
print(supplement['date'])

print("\nExperiments:")
experiments = pd.read_csv('experiments.csv')
print(experiments.dtypes)
print("\nexperiments dataframe:")
print(experiments)

print("\nUsers Profile:")
users = pd.read_csv('user_profiles.csv')
print(users.dtypes)
print("\nusers dataframe:")
print(users)

def merge_all_data(user_health_path, supplement_path, experiments_path, user_profile_path):
    # Read all CSV files
    user_health_df = pd.read_csv(user_health_path)
    supplement_df = pd.read_csv(supplement_path)
    experiments_df = pd.read_csv(experiments_path)
    user_profile_df = pd.read_csv(user_profile_path)
    
    # === PROCESS USER HEALTH ===
    user_health = user_health_df.copy()
    # Keep user_id as string (UUID format)
    user_health['user_id'] = user_health['user_id'].astype(str)
    # Convert date to datetime
    user_health['date'] = pd.to_datetime(user_health['date'])
    # Clean sleep_hours by removing 'h' or 'H' suffix and convert to float
    user_health['sleep_hours'] = user_health['sleep_hours'].astype(str).str.replace('h', '', case=False).str.strip().astype(float)
    # Convert numeric columns to float
    user_health['average_heart_rate'] = user_health['average_heart_rate'].astype(float)
    user_health['average_glucose'] = user_health['average_glucose'].astype(float)
    user_health['activity_level'] = user_health['activity_level'].astype('Int64')  # Nullable integer
    
    # === PROCESS SUPPLEMENT USAGE ===
    supplement = supplement_df.copy()
    supplement['user_id'] = supplement['user_id'].astype(str)
    supplement['date'] = pd.to_datetime(supplement['date'])
    # Convert dosage from mg to grams
    supplement['dosage_grams'] = supplement['dosage'] / 1000
    # Convert is_placebo to boolean
    supplement['is_placebo'] = supplement['is_placebo'].astype(bool)
    
    # === PROCESS EXPERIMENTS ===
    experiments = experiments_df.copy()
    experiments['experiment_id'] = experiments['experiment_id'].astype(str)
    # Rename 'name' column to 'experiment_name'
    experiments = experiments.rename(columns={'name': 'experiment_name'})
    
    # === PROCESS USER PROFILES ===
    user_profiles = user_profile_df.copy()
    user_profiles['user_id'] = user_profiles['user_id'].astype(str)
    # Keep email as string
    user_profiles['email'] = user_profiles['email'].astype(str)
    
    # Define function to categorize age groups
    def get_age_group(age):
        if pd.isna(age):
            return 'Unknown'
        elif age < 18:
            return 'Under 18'
        elif age <= 25:
            return '18-25'
        elif age <= 35:
            return '26-35'
        elif age <= 45:
            return '36-45'
        elif age <= 55:
            return '46-55'
        elif age <= 65:
            return '56-65'
        else:
            return 'Over 65'
    
    user_profiles['user_age_group'] = user_profiles['age'].apply(get_age_group)
    
    # === MERGE supplement with experiments to get experiment_name ===
    supplement_with_exp = pd.merge(
        supplement, 
        experiments[['experiment_id', 'experiment_name']], 
        on='experiment_id', 
        how='left'
    )
    
    # Select only the columns we need from supplement_with_exp
    supplement_with_exp = supplement_with_exp[['user_id', 'date', 'supplement_name', 'dosage_grams', 'is_placebo', 'experiment_name']]
    
    # === LEFT JOIN user_health with supplement data ===
    merged_df = pd.merge(
        user_health,
        supplement_with_exp,
        on=['user_id', 'date'],
        how='left'
    )
    
    # === LEFT JOIN with user_profiles to add demographic info ===
    merged_df = pd.merge(
        merged_df,
        user_profiles[['user_id', 'email', 'user_age_group']],
        on='user_id',
        how='left'
    )
    
    # === Fill missing values as specified ===
    # For days without supplement intake
    merged_df['supplement_name'] = merged_df['supplement_name'].fillna('No intake')
    merged_df['dosage_grams'] = merged_df['dosage_grams'].fillna(pd.NA)
    merged_df['is_placebo'] = merged_df['is_placebo'].fillna(pd.NA)
    merged_df['experiment_name'] = merged_df['experiment_name'].fillna(pd.NA)
    
    # Ensure boolean type for is_placebo (convert NA to None for proper boolean handling)
    merged_df['is_placebo'] = merged_df['is_placebo'].astype('boolean')  # Use pandas nullable boolean
    
    # Ensure date is datetime type
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    
    # Ensure email has no missing values (should not happen based on spec)
    merged_df['email'] = merged_df['email'].fillna('unknown@example.com')
    
    # Select final columns in the specified order
    final_columns = [
        'user_id',
        'date',
        'email',
        'user_age_group',
        'experiment_name',
        'supplement_name',
        'dosage_grams',
        'is_placebo',
        'average_heart_rate',
        'average_glucose',
        'sleep_hours',
        'activity_level'
    ]
    
    final_df = merged_df[final_columns].copy()
    
    # Final type conversions to match specifications
    final_df['user_id'] = final_df['user_id'].astype('string')
    final_df['date'] = pd.to_datetime(final_df['date'])
    final_df['email'] = final_df['email'].astype('string')
    final_df['user_age_group'] = final_df['user_age_group'].astype('string')
    final_df['experiment_name'] = final_df['experiment_name'].astype('string')
    final_df['supplement_name'] = final_df['supplement_name'].astype('string')
    final_df['dosage_grams'] = final_df['dosage_grams'].astype('float64')
    final_df['is_placebo'] = final_df['is_placebo'].astype('boolean')
    final_df['average_heart_rate'] = final_df['average_heart_rate'].astype('float64')
    final_df['average_glucose'] = final_df['average_glucose'].astype('float64')
    final_df['sleep_hours'] = final_df['sleep_hours'].astype('float64')
    final_df['activity_level'] = final_df['activity_level'].astype('Int64')
    
    return final_df


# def merge_all_data(user_health_path, supplement_path, experiments_path, user_profile_path):
#     # Read all CSV files
#     user_health_df = pd.read_csv(user_health_path)
#     supplement_df = pd.read_csv(supplement_path)
#     experiments_df = pd.read_csv(experiments_path)
#     user_profile_df = pd.read_csv(user_profile_path)
    
#     # === PROCESS USER HEALTH ===
#     user_health = user_health_df.copy()
#     # Convert date to datetime and handle the 'h' and 'H' suffixes in sleep_hours
#     user_health['date'] = pd.to_datetime(user_health['date'])
#     # Clean sleep_hours by removing 'h' or 'H' suffix and convert to float
#     user_health['sleep_hours'] = user_health['sleep_hours'].astype(str).str.replace('h', '', case=False).str.strip().astype(float)
    
#     # === PROCESS SUPPLEMENT USAGE ===
#     supplement = supplement_df.copy()
#     supplement['date'] = pd.to_datetime(supplement['date'])
#     # Convert dosage from mg to grams
#     supplement['dosage_grams'] = supplement['dosage'] / 1000
    
#     # === PROCESS EXPERIMENTS ===
#     experiments = experiments_df.copy()
    
#     # === PROCESS USER PROFILES ===
#     user_profiles = user_profile_df.copy()
#     # Define function to categorize age groups
#     def get_age_group(age):
#         if pd.isna(age):
#             return 'Unknown'
#         elif age < 18:
#             return 'Under 18'
#         elif age <= 25:
#             return '18-25'
#         elif age <= 35:
#             return '26-35'
#         elif age <= 45:
#             return '36-45'
#         elif age <= 55:
#             return '46-55'
#         elif age <= 65:
#             return '56-65'
#         else:
#             return 'Over 65'
    
#     user_profiles['user_age_group'] = user_profiles['age'].apply(get_age_group)
    
#     # === MERGE supplement with experiments to get experiment_name ===
#     supplement_with_exp = pd.merge(
#         supplement, 
#         experiments, 
#         on='experiment_id', 
#         how='left'
#     )
    
#     # === LEFT JOIN user_health with supplement data ===
#     # This ensures we keep all health records, even those without supplement intake
#     merged_df = pd.merge(
#         user_health,
#         supplement_with_exp,
#         on=['user_id', 'date'],
#         how='left'
#     )
    
#     # === LEFT JOIN with user_profiles to add demographic info ===
#     merged_df = pd.merge(
#         merged_df,
#         user_profiles[['user_id', 'email', 'user_age_group']],
#         on='user_id',
#         how='left'
#     )
    
#     # === Fill missing values as specified ===
#     # For days without supplement intake
#     merged_df['supplement_name'] = merged_df['supplement_name'].fillna('No intake')
#     merged_df['dosage_grams'] = merged_df['dosage_grams'].fillna(pd.NA)
#     merged_df['is_placebo'] = merged_df['is_placebo'].fillna(pd.NA)
#     merged_df['experiment_name'] = merged_df['name'].fillna(pd.NA)
    
#     # Drop the redundant 'name' column from experiments
#     if 'name' in merged_df.columns:
#         merged_df = merged_df.drop(columns=['name'])
    
#     # Ensure correct data types
#     merged_df['average_heart_rate'] = merged_df['average_heart_rate'].astype('float64')
#     merged_df['average_glucose'] = merged_df['average_glucose'].astype('float64')
#     merged_df['sleep_hours'] = merged_df['sleep_hours'].astype('float64')
#     merged_df['activity_level'] = merged_df['activity_level'].astype('Int64')
    
#     # === SELECT FINAL COLUMNS in the specified order ===
#     final_columns = [
#         'user_id',
#         'date',
#         'email',
#         'user_age_group',
#         'experiment_name',
#         'supplement_name',
#         'dosage_grams',
#         'is_placebo',
#         'average_heart_rate',
#         'average_glucose',
#         'sleep_hours',
#         'activity_level'
#     ]
    
#     final_df = merged_df[final_columns].copy()
    
#     # Convert date to date format (without time)
#     final_df['date'] = pd.to_datetime(final_df['date']).dt.date
    
#     return final_df