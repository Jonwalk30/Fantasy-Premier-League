import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from config import RAW_DATA_PATH, FEATURE_DATA, PREDICTIONS, TARGET_WEEKS_INTO_FUTURE, PARAMS, WEEKS_TO_TRUNCATE
import os


def truncate_previous_weeks(overall_data, weeks_to_truncate):
    all_data = overall_data.reset_index(['career_gw'])
    weeks_to_be_dropped = all_data.groupby(level=0).tail(weeks_to_truncate)
    weeks_to_be_dropped.set_index('career_gw', append=True, inplace=True)
    truncated_data = overall_data.drop(weeks_to_be_dropped.index)
    return truncated_data

def main():
    features = pd.read_csv(os.path.join(RAW_DATA_PATH, FEATURE_DATA),
                           index_col=['player', 'career_gw'])

    positions = features.element_type.unique()
    pred_pos_dict = {}

    for pos in positions:
        print(pos)
        # Model for each position
        players_in_pos = features[features['element_type'] == pos]

        # Train-test split
        if WEEKS_TO_TRUNCATE > 0:
            truncated_data = truncate_previous_weeks(overall_data=players_in_pos,
                                                     weeks_to_truncate=WEEKS_TO_TRUNCATE)
        else:
            truncated_data = players_in_pos.copy()

        grouped = truncated_data.reset_index(['career_gw'])

        non_train = grouped.groupby(level=0).tail(min(1, TARGET_WEEKS_INTO_FUTURE))
        non_train.set_index('career_gw', append=True, inplace=True)
        # test = players_in_pos.xs(players_in_pos.index.levels[1].max(), level=1, drop_level=False)
        # train = players_in_pos.drop(players_in_pos.index.levels[1].max(), level=1, axis=0)
        train = truncated_data.drop(non_train.index)
        test = grouped.groupby(level=0).tail(1)
        test.set_index('career_gw', append=True, inplace=True)


        # Drops rows with target NA which are not most recent month
        filtered_train = train[train.target.notna()].drop(['element_type', 'team',
                                                           'team_code', 'chance_of_playing_next_round'], axis=1)

        X_train_pd = filtered_train.drop('target', axis=1)
        y_train_pd = filtered_train['target']
        X_test_pd = test.drop(['target', 'element_type', 'team',
                               'team_code',  'chance_of_playing_next_round'], axis=1)

        # Normalize
        scaler = StandardScaler()
        scaler.fit(X_train_pd)
        X_train = scaler.transform(X_train_pd)
        y_train = y_train_pd.to_numpy()
        X_test = scaler.transform(X_test_pd)

        # Train model
        reg = ensemble.GradientBoostingRegressor(**PARAMS)
        reg.fit(X_train, y_train)

        # Predict
        pred = pd.DataFrame(reg.predict(X_test))
        pred.rename(columns={0: 'prediction'}, inplace=True)
        pred.index = test.index
        pred = pred.merge(test[['element_type', 'team', 'team_code',
                                'value_av_last_1_gws', 'chance_of_playing_next_round']],
                          left_index=True, right_index=True)
        pred.rename(columns={'value_av_last_1_gws': 'value'})
        print(pred.shape)
        pred_sorted = pred.sort_values(by='prediction', ascending=False)
        pred_sorted['pred_rank'] = pred_sorted['prediction'].rank(method='max', ascending=False)
        pred_pos_dict[pos] = pred_sorted
        print(pred_sorted.shape)

    output_preds = pd.concat(pred_pos_dict.values())
    output_preds.drop_duplicates().to_csv(os.path.join(RAW_DATA_PATH, PREDICTIONS), index=True)


if __name__ == "__main__":
    main()
