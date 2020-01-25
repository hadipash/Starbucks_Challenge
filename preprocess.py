import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split

data_dir = 'data'


def parse_offers(offers, transactions, name):
    data = []

    for _id, info in offers.items():
        history = transactions[(transactions['value'] == {'offer id': _id}) |
                               (transactions['value'] == {'offer_id': _id, 'reward': info['reward']})]
        if not history.empty:  # if the offer received
            receive_time = history[history['event'] == 'offer received']['time'].values
            viewed = history[history['event'] == 'offer viewed']
            completed = history[history['event'] == 'offer completed']
            trx_time = transactions[transactions['event'] == 'transaction']['time'].values

            value = 1
            for t in receive_time:
                view_time = viewed[(viewed['time'] >= t) &
                                   (viewed['time'] <= t + info['duration'] * 24)]['time'].to_list()
                if view_time:
                    # if viewed and responded during the influence time
                    if not completed[(completed['time'] >= view_time[0]) &
                                     (completed['time'] <= t + info['duration'] * 24)].empty or \
                            (name == 'info' and any(
                                view_time[0] <= tt <= t + info['duration'] * 24 for tt in trx_time)):
                        value = 3
                        break
                    # if viewed only during the influence time
                    else:
                        value = 2
            data.append({
                'value': value,
                'difficulty': info['difficulty'],
                'reward': info['reward'],
                'channels': info['channels']
            })

    return data


def transform(profile, transcript, offers):
    data = []

    for index, person in profile.iterrows():
        transactions = transcript[transcript['person'] == person['id']]
        amount_spend = transactions[transactions['event'] == 'transaction']['value'].to_list()
        amount_spend = sum([amount['amount'] for amount in amount_spend])

        for name, offer in offers.items():
            temp = parse_offers(offer, transactions, name)
            for t in temp:
                data.append({
                    'id': person['id'],
                    'gender': person['gender'],
                    'age': person['age'],
                    'income': person['income'],
                    'difficulty': t['difficulty'],
                    'reward': t['reward'],
                    'web': 1 if 'web' in t['channels'] else 0,          # email is not included (base case)
                    'mobile': 1 if 'mobile' in t['channels'] else 0,
                    'social': 1 if 'social' in t['channels'] else 0,
                    'bogo': t['value'] if name == 'bogo' else 0,
                    'disc': t['value'] if name == 'disc' else 0,
                    'info': t['value'] if name == 'info' else 0,
                    'amount_spend': amount_spend
                })

        if not (index + 1) % 1000:
            print('Processed {:d} entries'.format(index + 1))

    return data


def split_and_save(data, offers):
    # transform the gender column into separate columns ('Other' is the base case => no need a separate column)
    data = data.reindex(['F', 'M'] + list(data.columns.values), axis=1)
    data['F'] = data.apply(lambda x: 1 if x['gender'] == 'F' else 0, axis=1)
    data['M'] = data.apply(lambda x: 1 if x['gender'] == 'M' else 0, axis=1)
    data = data.drop(columns=['id', 'gender', 'amount_spend'])

    # splitting data into training, validation and test sets
    z = data.iloc[:, :-len(offers)]
    zz = data.iloc[:, -len(offers):]
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-len(offers)], data.iloc[:, -len(offers):],
                                                        test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    print('\nNumber of data points in train: {}, validation: {}, test: {}'.
          format(len(X_train.index), len(X_valid.index), len(X_test.index)))

    # Min-Max Feature scaling for 'age' and 'income'
    for x in [X_valid, X_test, X_train]:
        x['age'] = (x['age'] - X_train['age'].min()) / (X_train['age'].max() - X_train['age'].min())
        x['income'] = (x['income'] - X_train['income'].min()) / (X_train['income'].max() - X_train['income'].min())

    # Drop columns with unknown labels and save into files:
    for offer in offers:
        for key, _set in {'train': [X_train, y_train], 'valid': [X_valid, y_valid], 'test': [X_test, y_test]}.items():
            offer_y = _set[1][_set[1][offer] != 0][offer]
            offer_X = _set[0].loc[offer_y.index]

            offer_X.to_pickle(os.path.join(data_dir, offer + '_X_' + key + '.pkl'))
            offer_y.to_pickle(os.path.join(data_dir, offer + '_y_' + key + '.pkl'))


def preprocess(portfolio, profile, transcript):
    """
    Transforms given by Starbucks input files (portfolio.json, profile.json, transcript.json)
    into one big table suitable for future data analysis. Particularly, the new table looks as follows:
    | id | gender | age | income | offer_type1 | ... | offer_typeN | amount_spend |
    |----|--------|-----|--------|-------------|-----|-------------|--------------|
    The output table is saved as a pickle file (data/data.pkl)

    :param portfolio: DataFrame. portfolio.json converted to Pandas DataFrame
    :param profile: DataFrame. profile.json converted to Pandas DataFrame
    :param transcript: DataFrame. transcript.json converted to Pandas DataFrame
    """

    # read the offers from the files and convert into a dictionary
    # {'offer id':{'duration': int, 'reward': int, 'difficulty': int, 'channels': list}}
    bogo = portfolio[portfolio['offer_type'] ==
                     'bogo'][['id', 'duration', 'reward', 'difficulty', 'channels']]. \
                     set_index('id').to_dict(orient='index')
    disc = portfolio[portfolio['offer_type'] ==
                     'discount'][['id', 'duration', 'reward', 'difficulty', 'channels']]. \
                     set_index('id').to_dict(orient='index')
    info = portfolio[portfolio['offer_type'] ==
                     'informational'][['id', 'duration', 'reward', 'difficulty', 'channels']]. \
                     set_index('id').to_dict(orient='index')

    # Drop rows from the customer table with no information but a customer's id
    profile = profile.drop(profile[(profile['gender'].isna()) &
                                   (profile['age'] == 118) &
                                   (profile['income'].isna())].index).reset_index(drop=True)

    # apply data transformation
    data = transform(profile, transcript, {'bogo': bogo, 'disc': disc, 'info': info})
    data = pd.DataFrame(data, columns=['id', 'gender', 'age', 'income', 'difficulty', 'reward', 'web', 'mobile',
                                       'social', 'bogo', 'disc', 'info', 'amount_spend'])

    # apply grouping for easier data visualization
    data_brief = data.groupby(['id']).max().reset_index()
    data_brief.to_pickle(os.path.join(data_dir, 'data_brief.pkl'))
    split_and_save(data, ['bogo', 'disc', 'info'])


if __name__ == "__main__":
    start = time.time()
    portfolio = pd.read_json(os.path.join(data_dir, 'portfolio.json'), orient='records', lines=True)
    profile = pd.read_json(os.path.join(data_dir, 'profile.json'), orient='records', lines=True)
    transcript = pd.read_json(os.path.join(data_dir, 'transcript.json'), orient='records', lines=True)

    preprocess(portfolio, profile, transcript)
    print('Processing time {:.2f} s'.format(time.time() - start))
