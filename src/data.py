import numpy as np
import pandas as pd
from typing import Dict, Tuple
from geopy.distance import distance as geodistance
from .config import load_config
import re


# country name remaps
def remap_country_name_from_world_bank_to_main_df_name(country: str) -> str:
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

def remap_country_name_from_un_wpp_to_main_df_name(country: str) -> str:
    return {
        'Bahamas': 'The Bahamas',
        'Bolivia (Plurinational State of)': 'Bolivia',
        'Brunei Darussalam': 'Brunei',
        'China, Taiwan Province of China': 'Taiwan*',
        'Congo': 'Congo (Brazzaville)',
        "Côte d'Ivoire": "Cote d'Ivoire",
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

WB_CONV = {'Country Name': remap_country_name_from_world_bank_to_main_df_name}
UN_CONV = {'Location': remap_country_name_from_un_wpp_to_main_df_name}

# loaders
def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    train = pd.read_csv(cfg.paths['kaggle_train'], parse_dates=['Date'])
    test  = pd.read_csv(cfg.paths['kaggle_test'],  parse_dates=['Date'])
    return train, test

def _latest_value_by_year_cols(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    year_cols = [c for c in df.columns if c.isdigit()]
    df[value_name] = df[year_cols].apply(
        lambda row: row[row.last_valid_index()] if row.last_valid_index() else np.nan, axis=1
    )
    return df

def load_world_bank() -> Dict[str, pd.DataFrame]:
    cfg = load_config()
    area = pd.read_csv(cfg.paths['world_bank']['area'],    skiprows=4, converters=WB_CONV)
    smoking = pd.read_csv(cfg.paths['world_bank']['smoking'], skiprows=4, converters=WB_CONV)
    beds = pd.read_csv(cfg.paths['world_bank']['beds'],    skiprows=4, converters=WB_CONV)
    health = pd.read_csv(cfg.paths['world_bank']['health'],  skiprows=4, converters=WB_CONV)

    area   = _latest_value_by_year_cols(area,   'CountryArea')[['Country Name','CountryArea']]
    smoking= _latest_value_by_year_cols(smoking,'CountrySmokingRate')[['Country Name','CountrySmokingRate']]
    beds   = _latest_value_by_year_cols(beds,   'CountryHospitalBedsRate')[['Country Name','CountryHospitalBedsRate']]
    health = _latest_value_by_year_cols(health, 'CountryHealthExpenditurePerCapitaPPP')[['Country Name','CountryHealthExpenditurePerCapitaPPP']]
    return {'area': area, 'smoking': smoking, 'beds': beds, 'health': health}

def load_un_wpp_population() -> pd.DataFrame:
    cfg = load_config()
    df = pd.read_csv(
        cfg.paths['un_wpp_population'],
        usecols=['Location','Time','AgeGrp','PopMale','PopFemale','PopTotal'],
        parse_dates=['Time'],
        converters=UN_CONV
    )
    df = df[(df['Time'] >= pd.Timestamp(2014,1,1)) & (df['Time'] <= pd.Timestamp(2019,1,1))]

    agg_rows = []
    for (location, time), g in df.groupby(["Location","Time"]):
        buckets = [0]*5
        male = female = 0
        for _, r in g.iterrows():
            start = int(re.split(r"[\-\+]", r["AgeGrp"])[0])
            buckets[min(start // 20, 4)] += (r["PopMale"] + r["PopFemale"])
            male   += r["PopMale"]; female += r["PopFemale"]
        agg_rows.append({
            "Location": location, "Time": time,
            "CountryPop_0-20": buckets[0], "CountryPop_20-40": buckets[1],
            "CountryPop_40-60": buckets[2], "CountryPop_60-80": buckets[3],
            "CountryPop_80+": buckets[4],
            "CountryPopMale": male, "CountryPopFemale": female,
            "CountryPopTotal": male + female,
        })
    agg = pd.DataFrame(agg_rows).sort_values('Time').drop_duplicates(['Location'], keep='last')
    return agg.drop(columns='Time')

# feature engineering 
def _get_hubei_coords(df: pd.DataFrame) -> Tuple[float, float]:
    h = df[df['Province/State'] == 'Hubei']
    if h.empty:
        raise RuntimeError('Hubei not found in data')
    r = h.iloc[0]
    return float(r['Lat']), float(r['Long'])

def _is_cumulative(incr: np.ndarray) -> bool:
    # cumulated series must not have negative daily increments
    return np.all(np.isnan(incr) | (incr >= 0))

def build_main_df() -> pd.DataFrame:
    cfg = load_config()
    train, test = load_raw()

    last_train_date = train['Date'].max()
    test_wo_train = test.drop(index=test[test['Date'] <= last_train_date].index)

    main_df = pd.concat([train, test_wo_train], ignore_index=True)
    from_ships = main_df['Province/State'].isin(['From Diamond Princess','Grand Princess'])
    main_df.loc[from_ships, ['Province/State','Country/Region']] = (
        main_df.loc[from_ships, ['Country/Region','Province/State']].values
    )

    # sort & fill
    main_df.sort_values(by='Date', inplace=True)
    for c in ['Country/Region','Province/State']:
        main_df[c] = main_df[c].fillna('')

    # time-delay embedding
    days_history = int(cfg.modelling['days_history_size'])

    # init kolumn
    for fld in ['LogNewConfirmedCases', 'LogNewFatalities']:
        main_df[fld] = np.nan
        for k in range(1, days_history + 1):
            main_df[f"{fld}_prev_day_{k}"] = np.nan

    bad_idx = []  # zbierz indeksy do usunięcia po pętli

    for keys, loc_df in main_df.groupby(['Country/Region', 'Province/State'], sort=False):
        idx = loc_df.index

        # sprawdź kumulatywność dla obu serii naraz
        is_ok = True
        for field in ['ConfirmedCases', 'Fatalities']:
            vals = loc_df[field].astype(float).to_numpy()
            incr = np.diff(np.r_[0.0, vals])  # dzienne przyrosty
            if np.any(incr[~np.isnan(incr)] < 0):
                is_ok = False
                break

        if not is_ok:
            bad_idx.append(idx)
            continue  # nie licz cech dla wadliwych grup

        # oblicz cechy dla obu targetów
        for field in ['ConfirmedCases', 'Fatalities']:
            vals = loc_df[field].astype(float).to_numpy()
            incr = np.diff(np.r_[0.0, vals])
            log_new = np.log1p(np.maximum(incr, 0.0))
            main_df.loc[idx, 'LogNew' + field] = log_new
            for k in range(1, days_history + 1):
                main_df.loc[idx[k:], f'LogNew{field}_prev_day_{k}'] = log_new[:-k]

    # dopiero teraz usuń złe lokalizacje
    if bad_idx:
        bad_idx = np.concatenate([i.values if hasattr(i, "values") else np.array(i) for i in bad_idx])
        main_df = main_df.drop(index=bad_idx).copy()

    # Day / WeekDay
    first_date = main_df['Date'].min()
    main_df['Day'] = (main_df['Date'] - first_date).dt.days.astype('int32')
    main_df['WeekDay'] = main_df['Date'].dt.weekday

    # Days-since thresholds
    for th in cfg.modelling['thresholds_since']:
        main_df[f'Days_since_ConfirmedCases={th}'] = np.nan
        main_df[f'Days_since_Fatalities={th}']      = np.nan
    for (_, _), loc_df in main_df.groupby(['Country/Region','Province/State']):
        g = loc_df.sort_values('Date')
        for field in ['ConfirmedCases','Fatalities']:
            for th in cfg.modelling['thresholds_since']:
                first_day = g['Day'].loc[g[field] >= th].min()
                if pd.notna(first_day):
                    main_df.loc[g.index, f'Days_since_{field}={th}'] = g['Day'].apply(
                        lambda d: -1 if d < first_day else d - first_day
                    ).values

    # Distance to origin (Hubei)
    origin = _get_hubei_coords(main_df)
    main_df['Distance_to_origin'] = main_df.apply(
        lambda r: geodistance((r['Lat'], r['Long']), origin).km, axis=1
    )

    # external features: World Bank + UN WPP
    wb = load_world_bank()
    def _merge_drop(left: pd.DataFrame, right: pd.DataFrame, right_col='Country Name'):
        df = left.merge(right, how='left', left_on='Country/Region', right_on=right_col)
        if right_col in df.columns: df.drop(columns=[right_col], inplace=True)
        return df
    for key in ['area','smoking','beds','health']:
        main_df = _merge_drop(main_df, wb[key], right_col='Country Name')

    pop = load_un_wpp_population()
    main_df = _merge_drop(main_df, pop, right_col='Location')

    # density
    if 'CountryPopTotal' in main_df and 'CountryArea' in main_df:
        main_df['CountryPopDensity'] = main_df['CountryPopTotal'] / main_df['CountryArea']

    return main_df

def make_splits(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    cfg = load_config()
    d_train = pd.Timestamp(cfg.dates['last_train_date'])
    d_eval  = pd.Timestamp(cfg.dates['last_eval_date'])
    train = df[df['Date'] <= d_train].copy()
    eval_ = df[(df['Date'] > d_train) & (df['Date'] <= d_eval)].copy()
    test  = df[df['Date'] > d_eval].copy()
    return {'train': train, 'eval': eval_, 'test': test}

def preprocess_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    labels = df[['LogNewConfirmedCases','LogNewFatalities']].copy()
    features_df = df.drop(columns=[c for c in cfg.features['drop_columns'] if c in df.columns]).copy()
    return features_df, labels