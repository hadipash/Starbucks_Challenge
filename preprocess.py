"""
This file contains cleaning, preprocessing and splitting data helper functions for further use in analysis.ipynb
"""

import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_offers(offers, transactions, offer_name):
    """
    Extracts the responses to offers given by each customer
    
    :param offers: dict. All offers of single type (bogo, or disc, or info)
    :param transactions: DataFrame. Transaction made by a customer
    :param offer_name: str. Name of an offer type (bogo, or disc, or info)
    :return: list of responses given by a customer to offers
    """
    data = []

    for _id, info in offers.items():
        # get all the transactions associated with an offer
        history = transactions[(transactions['value'] == {'offer id': _id}) |
                               (transactions['value'] == {'offer_id': _id, 'reward': info['reward']})]
        if not history.empty:  # if the offer received
            receive_time = history[history['event'] == 'offer received']['time'].values
            viewed = history[history['event'] == 'offer viewed']
            completed = history[history['event'] == 'offer completed']
            trx_time = transactions[transactions['event'] == 'transaction']['time'].values

            response = 1
            # check whether the offer was viewed and completed within the influence period specified by Starbucks
            for rt in receive_time:
                view_time = viewed[(viewed['time'] >= rt) &
                                   (viewed['time'] <= rt + info['duration'] * 24)]['time'].to_list()
                if view_time:
                    # if viewed and responded during the influence period
                    if not completed[(completed['time'] >= view_time[0]) &
                                     (completed['time'] <= rt + info['duration'] * 24)].empty or \
                            (offer_name == 'info' and any(
                                view_time[0] <= tt <= rt + info['duration'] * 24 for tt in trx_time)):
                        response = 3
                        break   # only the best response is recorded
                    # if only viewed during the influence period
                    else:
                        response = 2
            data.append({
                'response': response,
                'difficulty': info['difficulty'],
                'reward': info['reward'],
                'channels': info['channels']
            })

    return data


def transform(profile, transcript, offers):
    """
    Transforms given by Starbucks input files into one big table suitable for future data analysis.
    Particularly, the new table looks as follows:
    | id | gender | age | income | difficulty | reward | web | mobile | social | bogo | disc | info | amount_spend |
    |----|--------|-----|--------|------------|--------|-----|--------|--------|------|------|------|--------------|

    :param profile: DataFrame. profile.json converted to Pandas DataFrame
    :param transcript: DataFrame. transcript.json converted to Pandas DataFrame
    :param offers: dict. Contains information about each offer type
    :return: list of dictionaries, each of which corresponds to a row in the final table.
    """
    data = []

    # iterate over each person's information and construct the table with responses given by customers
    # the table may contain rows with a person several times, each corresponds to a different response to an offer
    for index, person in profile.iterrows():
        transactions = transcript[transcript['person'] == person['id']]
        amount_spend = transactions[transactions['event'] == 'transaction']['value'].to_list()
        amount_spend = sum([amount['amount'] for amount in amount_spend])

        for name, offer in offers.items():
            parsed = parse_offers(offer, transactions, name)
            for po in parsed:
                data.append({
                    'id': person['id'],
                    'gender': person['gender'],
                    'age': person['age'],
                    'income': person['income'],
                    'difficulty': po['difficulty'],
                    'reward': po['reward'],
                    'web': 1 if 'web' in po['channels'] else 0,          # email column is not included (base case)
                    'mobile': 1 if 'mobile' in po['channels'] else 0,
                    'social': 1 if 'social' in po['channels'] else 0,
                    'bogo': po['response'] if name == 'bogo' else 0,
                    'disc': po['response'] if name == 'disc' else 0,
                    'info': po['response'] if name == 'info' else 0,
                    'amount_spend': amount_spend
                })

        if not (index + 1) % 1000:
            print('Processed {:d} entries'.format(index + 1))

    return data


def preprocess(portfolio, profile, transcript, save_dir='data'):
    """
    Preprocesses, cleans the input data and then transforms several input sets into one.
    The output table is saved as pickle files in 2 forms: first is for feeding into algorithms (data.pkl) and
    the other one (with no duplication by id) is for visualization (data_brief.pkl)

    :param portfolio: DataFrame. portfolio.json converted to Pandas DataFrame
    :param profile: DataFrame. profile.json converted to Pandas DataFrame
    :param transcript: DataFrame. transcript.json converted to Pandas DataFrame
    :param save_dir: str. Output files saving location
    """

    # read offers from the files and convert into a dictionary
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

    # save preprocessed data
    data.to_pickle(os.path.join(save_dir, 'data.pkl'))
    print('Saving data.pkl')

    # apply grouping for easier data visualization and save
    data_brief = data.drop(columns=['difficulty', 'reward'])
    data_brief = data_brief.groupby(['id']).max().reset_index()
    data_brief.to_pickle(os.path.join(save_dir, 'data_brief.pkl'))
    print('Saving data_brief.pkl')


def split_data(data, offers, stratify=None, scaling=None):
    """
    Splits data into train, validation and test sets.

    :param data: DataFrame. Data to be split into sets
    :param offers: list. Names of offer types
    :param stratify: list. Column names to be used for data split in stratified fashion
    :param scaling: list. Column names to apply Min-Max Scaling on
    :return: dict. Features and labels for each set and offer type.
    """
    # avoid Mutable Default Argument
    if scaling is None:
        scaling = []

    # create a DataFrame with unique customer ids to avoid the same person appearing in different sets
    data_brief = data.groupby(['id']).max().reset_index()

    # splitting data into training, validation and test sets
    train_ids, test_ids = train_test_split(data_brief, stratify=data_brief[stratify] if stratify else None,
                                           test_size=0.2, random_state=42)
    train_ids, valid_ids = train_test_split(train_ids, stratify=train_ids[stratify] if stratify else None,
                                            test_size=0.2, random_state=42)

    # transform the gender column into separate columns ('Other' is the base case => no need a separate column)
    data = data.reindex(['F', 'M'] + list(data.columns.values), axis=1)
    data['F'] = data.apply(lambda x: 1 if x['gender'] == 'F' else 0, axis=1)
    data['M'] = data.apply(lambda x: 1 if x['gender'] == 'M' else 0, axis=1)
    data = data.drop(columns=['gender', 'amount_spend'])

    # create sets with no same person (id) appearing in different sets
    train = data[data['id'].isin(train_ids['id'])]
    valid = data[data['id'].isin(valid_ids['id'])]
    test = data[data['id'].isin(test_ids['id'])]

    X_train = train.iloc[:, :-len(offers)].drop(columns=['id'])
    y_train = train.iloc[:, -len(offers):]
    X_valid = valid.iloc[:, :-len(offers)].drop(columns=['id'])
    y_valid = valid.iloc[:, -len(offers):]
    X_test = test.iloc[:, :-len(offers)].drop(columns=['id'])
    y_test = test.iloc[:, -len(offers):]

    print('Number of data points in train: {}, validation: {}, test: {}'.
          format(len(X_train.index), len(X_valid.index), len(X_test.index)))

    # Min-Max Feature scaling
    for feature in scaling:
        for x in [X_valid, X_test, X_train]:
            x[feature] = (x[feature] - X_train[feature].min()) / (X_train[feature].max() - X_train[feature].min())

    # Drop rows with unknown labels (0) and prepare sets for convenient access:
    data = {}
    for offer in offers:
        data.update({offer: {}})
        for key, _set in {'train': [X_train, y_train], 'valid': [X_valid, y_valid], 'test': [X_test, y_test]}.items():
            y = _set[1][_set[1][offer] != 0][offer]
            X = _set[0].loc[y.index]

            data[offer].update({key: {}})
            data[offer][key]['X'] = X
            data[offer][key]['y'] = y

    return data


if __name__ == "__main__":
    start = time.time()
    portfolio = pd.read_json(os.path.join('data', 'portfolio.json'), orient='records', lines=True)
    profile = pd.read_json(os.path.join('data', 'profile.json'), orient='records', lines=True)
    transcript = pd.read_json(os.path.join('data', 'transcript.json'), orient='records', lines=True)

    preprocess(portfolio, profile, transcript)
    print('Preprocessing time {:.2f} s'.format(time.time() - start))
