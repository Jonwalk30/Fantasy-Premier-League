import pandas as pd
import numpy as np
import os
import pandas_gbq
from config import *


def create_index(df):
    df['career_gw'] = df['season'] + " | " + df['GW'].astype(int).astype(str).str.pad(width=2, fillchar='0')
    return df.set_index(['player', 'career_gw']).sort_index(level=[0, 1])


def add_own_team_features(df):
    df['team_goals_scored'] = np.where(df['was_home'] is True, df['team_h_score'], df['team_a_score'])
    df['team_points'] = np.where(df['was_home'] is True,
                                 np.where(df['team_h_score'] > df['team_a_score'],
                                          3,
                                          np.where(df['team_h_score'] == df['team_a_score'],
                                                   1,
                                                   0)
                                          ),
                                 np.where(df['team_h_score'] > df['team_a_score'],
                                          0,
                                          np.where(df['team_h_score'] == df['team_a_score'],
                                                   1,
                                                   3)
                                          )
                                 )
    return df


def create_target(gw_df, target_type, target_weeks_into_future):
  # Positive integer -> we will take an average of scores over that many weeks in subsequent gameweeks
  if target_type.upper() == "AVG" and isinstance(target_weeks_into_future, int) and target_weeks_into_future > 0:
    target = gw_df.groupby(level=0)['total_points'].shift(0).rolling(target_weeks_into_future).mean().shift(-target_weeks_into_future)
  # 0 or negative -> we will take the exponential average of all subsequent gameweek scores
  elif target_type.upper() == "EWM":
    target = gw_df.groupby(level=0)['total_points'].shift(-1).sort_index(ascending=False).shift(0).ewm(com=1).mean().sort_index(ascending=True)
  # Default to target being the next gameweek's scpre
  else:
    target = gw_df.groupby(level=0)['total_points'].shift(-1)
  return target


def create_feature_over_time(time_features, past_weeks_num, features_df, base_features_df):
    for feat in time_features:
        for x in past_weeks_num:
            post_pend = f"_av_last_{x}_gws"
            features_df[feat + post_pend] = base_features_df.groupby(level=0)[feat].shift(0).rolling(x).mean()
            post_pend = f"_diff_last_{x}_gws"
            features_df[feat + post_pend] = base_features_df.groupby(
                level=0)[feat].shift(0) - base_features_df.groupby(level=0)[feat].shift(x)
        post_pend = "_ewm"
        features_df[feat + post_pend] = base_features_df.groupby(level=0)[feat].shift(0).ewm(com=1).mean()
        post_pend = "_atm"
        features_df[feat + post_pend] = base_features_df.groupby(level=0)[feat].shift(0).expanding().mean()
    return features_df


def main():

    gw_df = pandas_gbq.read_gbq(' SELECT * FROM fpl_staging_data.' + INGESTED_DATA, project_id=PROJECT_ID).drop_duplicates()

    gw_df_team_features = gw_df.pipe(create_index).pipe(add_own_team_features)

    gw_df_team_features['is_home'] = gw_df_team_features['was_home']

    gw_df_filtered = gw_df_team_features[TIME_RELATED_FEATURES + ['is_home', 'element_type', 'team', 'team_code']]

    features_df = gw_df_filtered[['element_type', 'is_home', 'team', 'team_code']]

    features_df['target'] = create_target(gw_df_filtered, TARGET_TYPE, TARGET_WEEKS_INTO_FUTURE)

    features_with_time_df = create_feature_over_time(time_features=TIME_RELATED_FEATURES, past_weeks_num=PAST_WEEKS_NUM,
                                                     features_df=features_df, base_features_df=gw_df_filtered)

    for col in features_with_time_df.columns.difference(['target']):
        features_with_time_df[col] = features_with_time_df[col].fillna(0).replace([np.inf, -np.inf], 100)

    features_with_time_df = features_with_time_df.drop('is_home', axis=1)
    features_with_time_df = features_with_time_df.reset_index()
    features_with_time_df.drop_duplicates().to_gbq(destination_table = 'fpl_staging_data.'+FEATURE_DATA, if_exists="replace")


if __name__ == "__main__":
    main()
