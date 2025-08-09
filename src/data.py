import os
import re
import warnings
import numpy as np
import pandas as pd
import geopy.distance

# Keep original imports to minimize diffs (even if unused)
import sklearn.preprocessing  # noqa: F401
import catboost as cb         # noqa: F401
import xgboost as xgb         # noqa: F401
import tensorflow as tf       # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
import seaborn as sns            # noqa: F401

from .config import (
    paths, last_train_date, last_eval_date, last_test_date,
    cat_features, drop_columns, days_history_size, thresholds
)

warnings.filterwarnings("ignore")

location_columns = ['Country/Region','Province/State']

# ---- Helpers (names preserved) ----
def is_cumulative(increment_series):
    for v in increment_series:
        if (not np.isnan(v)) and (v < 0):
            return False
    return True

def get_hubei_coords(df):
    for _, row in df.iterrows():
        if row['Province/State'] == 'Hubei':
            return (row['Lat'], row['Long'])
    raise Exception('Hubei not found in data')

def merge_with_column_drop(left_df, right_df, right_df_column='Country'):
    df = pd.merge(
        left=left_df,
        right=right_df,
        how='left',
        left_on='Country/Region',
        right_on=right_df_column
    )
    df.drop(columns=right_df_column, inplace=True)
    return df

def remap_country_name_from_world_bank_to_main_df_name(country):
    return {
        'Bahamas, The': 'The Bahamas',
        'Brunei Darussalam': 'Brunei',
        'Congo, Rep.': 'Congo (Brazzaville)',
        'Congo, Dem. Rep.': 'Congo (Kinshasa)',
        'Czech Republic': 'Czechia',
        'Egypt, Arab Rep.': 'Egypt',
        'Iran, Islamic Rep.': 'Iran',
        'Korea, Rep.': 'Korea, South',
        'Kyrgyz Republic': 'Kyrgyzstan',
        'Russian Federation': 'Russia',
        'Slovak Republic': 'Slovakia',
        'St. Lucia': 'Saint Lucia',
        'St. Vincent and the Grenadines': 'Saint Vincent and the Grenadines',
        'United States': 'US',
        'Venezuela, RB': 'Venezuela',
    }.get(country, country)

def remap_country_name_from_un_wpp_to_main_df_name(country):
    return {
        'Bahamas': 'The Bahamas',
        'Bolivia (Plurinational State of)': 'Bolivia',
        'Brunei Darussalam': 'Brunei',
        'China, Taiwan Province of China': 'Taiwan*',
        'Congo' : 'Congo (Brazzaville)',
        'Côte d\'Ivoire': 'Cote d\'Ivoire',
        'Democratic Republic of the Congo': 'Congo (Kinshasa)',
        'Gambia': 'The Gambia',
        'Iran (Islamic Republic of)': 'Iran',
        'Republic of Korea': 'Korea, South',
        'Republic of Moldova': 'Moldova',
        'Réunion': 'Reunion',
        'Russian Federation': 'Russia',
        'United Republic of Tanzania': 'Tanzania',
        'United States of America': 'US',
        'Venezuela (Bolivarian Republic of)': 'Venezuela',
        'Viet Nam': 'Vietnam'
    }.get(country, country)

world_bank_converters={'Country Name': remap_country_name_from_world_bank_to_main_df_name}
un_wpp_converters={'Location': remap_country_name_from_un_wpp_to_main_df_name}

def preprocess_df(df):
    labels = df[['LogNewConfirmedCases', 'LogNewFatalities']].copy()
    features_df = df.drop(columns=drop_columns).copy()
    return features_df, labels

# ---- Main data pipeline (names preserved) ----
def build_datasets():
    # Load
    original_train_df = pd.read_csv(paths["kaggle_train"], parse_dates=['Date'])
    original_test_df  = pd.read_csv(paths["kaggle_test"],  parse_dates=['Date'])

    last_original_train_date = original_train_df['Date'].max()
    original_test_wo_train_df = original_test_df.drop(
        index=original_test_df[original_test_df['Date'] <= last_original_train_date].index
    )
    main_df = pd.concat([original_train_df, original_test_wo_train_df], ignore_index=True)

    # Cruise ships swap
    from_cruise_ships = main_df['Province/State'].isin(['From Diamond Princess', 'Grand Princess'])
    main_df.loc[from_cruise_ships, ['Province/State','Country/Region']] = \
        main_df.loc[from_cruise_ships, ['Country/Region','Province/State']].values

    # Feature engineering
    main_df.sort_values(by='Date', inplace=True)

    for column in location_columns:
        main_df[column].fillna('', inplace=True)

    print('data size before removing bad data = ', len(main_df))
    for field in ['LogNewConfirmedCases', 'LogNewFatalities']:
        main_df[field] = np.nan
        for prev_day in range(1, days_history_size + 1):
            main_df[field + '_prev_day_%s' % prev_day] = np.nan

    for location_name, location_df in main_df.groupby(location_columns):
        for field in ['ConfirmedCases', 'Fatalities']:
            new_values = location_df[field].values.copy()
            new_values[1:] -= new_values[:-1]
            if not is_cumulative(new_values):
                print('%s for %s, %s is not valid cumulative series, drop it' % ((field,) + location_name))
                main_df.drop(index=location_df.index, inplace=True)
                break
            log_new_values = np.log1p(new_values)
            main_df.loc[location_df.index, 'LogNew' + field] = log_new_values
            for prev_day in range(1, days_history_size + 1):
                main_df.loc[location_df.index[prev_day:], 'LogNew%s_prev_day_%s' % (field, prev_day)] = (
                    log_new_values[:-prev_day]
                )
    print('data size after removing bad data = ', len(main_df))

    # Day and WeekDay
    first_date = min(main_df['Date'])
    main_df['Day'] = (main_df['Date'] - first_date).dt.days.astype('int32')
    main_df['WeekDay'] = main_df['Date'].transform(lambda d: d.weekday())

    # Days since Xth
    for threshold in thresholds:
        main_df['Days_since_ConfirmedCases=%s' % threshold] = np.nan
        main_df['Days_since_Fatalities=%s' % threshold] = np.nan

    for location_name, location_df in main_df.groupby(location_columns):
        for field in ['ConfirmedCases', 'Fatalities']:
            for threshold in thresholds:
                first_day = location_df['Day'].loc[location_df[field] >= threshold].min()
                if not np.isnan(first_day):
                    main_df.loc[location_df.index, 'Days_since_%s=%s' % (field, threshold)] = \
                        location_df['Day'].transform(lambda day: -1 if (day < first_day) else (day - first_day))

    # Distance to origin (Hubei)
    origin_coords = get_hubei_coords(main_df)
    main_df['Distance_to_origin'] = main_df.apply(
        lambda row: geopy.distance.distance((row['Lat'], row['Long']), origin_coords).km,
        axis='columns'
    )

    # External merges
    area_df = pd.read_csv(paths["world_bank"]["area"], skiprows=4, converters=world_bank_converters)
    year_columns = [str(year) for year in range(1960, 2020)]
    area_df['CountryArea'] = area_df[year_columns].apply(
        lambda row: row[row.last_valid_index()] if row.last_valid_index() else np.nan,
        axis='columns'
    )
    area_df = area_df[['Country Name', 'CountryArea']]
    main_df = merge_with_column_drop(main_df, area_df, right_df_column='Country Name')

    population_df = pd.read_csv(
        paths["un_wpp_population"],
        usecols=['Location', 'Time', 'AgeGrp', 'PopMale', 'PopFemale', 'PopTotal'],
        parse_dates=['Time'],
        converters= {'Location': remap_country_name_from_un_wpp_to_main_df_name}
    )
    population_df = population_df.loc[
        (population_df['Time'] >= pd.Timestamp(2014,1,1))
        & (population_df['Time'] <= pd.Timestamp(2019,1,1))
    ]

    aggregated_population_df = pd.DataFrame()
    for (location, time), group_df in population_df.groupby(["Location", "Time"]):
        pop_by_age_groups = [0]*5
        pop_male = 0
        pop_female = 0
        for _, row in group_df.iterrows():
            age_grp_start = int(re.split(r"[\-\+]", row["AgeGrp"])[0])
            pop_by_age_groups[min(age_grp_start // 20, 4)] += (row["PopMale"] + row["PopFemale"])
            pop_male += row["PopMale"]
            pop_female += row["PopFemale"]
        new_row = pd.DataFrame([{
            "Location": location,
            "Time": time,
            "CountryPop_0-20": pop_by_age_groups[0],
            "CountryPop_20-40": pop_by_age_groups[1],
            "CountryPop_40-60": pop_by_age_groups[2],
            "CountryPop_60-80": pop_by_age_groups[3],
            "CountryPop_80+":   pop_by_age_groups[4],
            "CountryPopMale":   pop_male,
            "CountryPopFemale": pop_female,
            "CountryPopTotal":  pop_male + pop_female,
        }])
        aggregated_population_df = pd.concat([aggregated_population_df, new_row], ignore_index=True)

    aggregated_population_df = aggregated_population_df.sort_values('Time').drop_duplicates(['Location'], keep='last')
    aggregated_population_df.drop(columns='Time', inplace=True)
    main_df = merge_with_column_drop(main_df, aggregated_population_df, right_df_column='Location')

    main_df['CountryPopDensity'] = main_df['CountryPopTotal'] / main_df['CountryArea']

    recent_year_columns = [str(year) for year in range(2010, 2020)]

    smoking_df = pd.read_csv(paths["world_bank"]["smoking"], skiprows=4, converters=world_bank_converters)
    smoking_df['CountrySmokingRate'] = smoking_df[recent_year_columns].apply(
        lambda row: row[row.last_valid_index()] if row.last_valid_index() else np.nan,
        axis='columns'
    )
    smoking_df = smoking_df[['Country Name', 'CountrySmokingRate']]
    main_df = merge_with_column_drop(main_df, smoking_df, right_df_column='Country Name')

    hospital_beds_df = pd.read_csv(paths["world_bank"]["beds"], skiprows=4, converters=world_bank_converters)
    hospital_beds_df['CountryHospitalBedsRate'] = hospital_beds_df[recent_year_columns].apply(
        lambda row: row[row.last_valid_index()] if row.last_valid_index() else np.nan,
        axis='columns'
    )
    hospital_beds_df = hospital_beds_df[['Country Name', 'CountryHospitalBedsRate']]
    main_df = merge_with_column_drop(main_df, hospital_beds_df, right_df_column='Country Name')

    health_expenditure_df = pd.read_csv(paths["world_bank"]["health"], skiprows=4, converters=world_bank_converters)
    health_expenditure_df['CountryHealthExpenditurePerCapitaPPP'] = health_expenditure_df[recent_year_columns].apply(
        lambda row: row[row.last_valid_index()] if row.last_valid_index() else np.nan,
        axis='columns'
    )
    health_expenditure_df = health_expenditure_df[['Country Name', 'CountryHealthExpenditurePerCapitaPPP']]
    main_df = merge_with_column_drop(main_df, health_expenditure_df, right_df_column='Country Name')

    # Splits (dates from YAML)
    train_df = main_df[main_df['Date'] <= last_train_date].copy()
    eval_df  = main_df[(main_df['Date'] > last_train_date) & (main_df['Date'] <= last_eval_date)].copy()
    test_df  = main_df[main_df['Date'] > last_eval_date].copy()

    train_features_df, train_labels = preprocess_df(train_df)
    eval_features_df,  eval_labels  = preprocess_df(eval_df)
    test_features_df,  _            = preprocess_df(test_df)

    return {
        "main_df": main_df,
        "train_df": train_df,
        "eval_df": eval_df,
        "test_df": test_df,
        "train_features_df": train_features_df,
        "train_labels": train_labels,
        "eval_features_df": eval_features_df,
        "eval_labels": eval_labels,
        "test_features_df": test_features_df,
        "location_columns": location_columns
    }