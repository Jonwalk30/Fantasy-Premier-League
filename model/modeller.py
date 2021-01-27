import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from config import *
import os


def main():
    features = pandas_gbq.read_gbq(' SELECT * FROM fpl_staging_data.' + FEATURE_DATA, project_id=PROJECT_ID).drop_duplicates()

    features = features.set_index(['player', 'career_gw'])
    params = {'n_estimators': 25,
              'max_depth': 2,
              'min_samples_split': 5,
              'learning_rate': 0.1,
              'loss': 'ls'}
    players = pd.DataFrame(columns=['player', 'prediction', 'position'])
    positions = [1, 2, 3, 4]
    for pos in positions:
        print(pos)
        # Model for each position
        players_in_pos = features[features['element_type'] == pos].drop('element_type', axis=1)

        # Train-test split
        X = players_in_pos.drop('target', axis=1)
        y = players_in_pos['target']
        X_train = X.groupby(level=0).apply(lambda x: x.iloc[:-1])
        X_test = X.groupby(level=0).tail(1)
        y_train = y.groupby(level=0).apply(lambda x: x.iloc[:-1])

        # Normalize
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        y_train = y_train.to_numpy()
        X_test = scaler.transform(X_test)

        # Train model
        reg = ensemble.GradientBoostingRegressor(**params)
        reg.fit(X_train, y_train)

        # Predict
        pred = pd.DataFrame(reg.predict(X_test))
        pred['player'] = X.groupby(level=0).tail(1).reset_index()['player']
        pred['position'] = pos
        pred.rename(columns={0: 'prediction'}, inplace=True)
        players = players.append(pred)

    players = players.sort_values(by='prediction', ascending=False)
    players['pred_rank'] = players['prediction'].rank(method='max', ascending=False)
    players = players[['player', 'prediction', 'position']]
    
    players.drop_duplicates().to_gbq(destination_table = 'fpl_staging_data.'+PREDICTIONS, if_exists="replace")


if __name__ == "__main__":
    main()
