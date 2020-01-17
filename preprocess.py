import time
import pandas as pd


def parse_offers(offers, data, transactions, column):
    for _id, info in offers.items():
        if data[column] == 3:   # if a customer has responded to the offer already
            break

        history = transactions[(transactions['value'] == {'offer id': _id}) |
                               (transactions['value'] == {'offer_id': _id, 'reward': info['reward']})]
        if not history.empty:  # if the offer received
            receive_time = history[history['event'] == 'offer received']['time'].values
            viewed = history[history['event'] == 'offer viewed']
            completed = history[history['event'] == 'offer completed']
            trx_time = transactions[transactions['event'] == 'transaction']['time'].values

            for t in receive_time:
                view_time = viewed[(viewed['time'] >= t) &
                                   (viewed['time'] <= t + info['duration'] * 24)]['time'].to_list()
                if view_time:
                    # if viewed and responded during the influence time
                    if not completed[(completed['time'] >= view_time[0]) &
                                     (completed['time'] <= t + info['duration'] * 24)].empty or \
                            (column == 'info' and any(view_time[0] <= tt <= t + info['duration'] * 24 for tt in trx_time)):
                        data[column] = 3
                        break
                    # if viewed only during the influence time
                    else:
                        data[column] = 2
                # if received only
                elif data[column] < 2:
                    data[column] = 1


def transform(person, transcript, offers):
    transactions = transcript[transcript['person'] == person['id']]
    for name, offer in offers.items():
        parse_offers(offer, person, transactions, name)

    amount_spend = transactions[transactions['event'] == 'transaction']['value'].to_list()
    amount_spend = sum([amount['amount'] for amount in amount_spend])
    person['amount_spend'] = amount_spend

    return person


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
    # read the offers from the files and convert into a dictionary {'offer id': {'duration': N, 'reward': N}}
    bogo = portfolio[portfolio['offer_type'] == 'bogo'][['id', 'duration', 'reward']]. \
        set_index('id').to_dict(orient='index')
    disc = portfolio[portfolio['offer_type'] == 'discount'][['id', 'duration', 'reward']]. \
        set_index('id').to_dict(orient='index')
    info = portfolio[portfolio['offer_type'] == 'informational'][['id', 'duration', 'reward']]. \
        set_index('id').to_dict(orient='index')

    # Drop rows from the customer table with no information but a customer's id
    profile = profile.drop(profile[(profile['gender'].isna()) &
                                   (profile['age'] == 118) &
                                   (profile['income'].isna())].index).reset_index(drop=True)

    # dataframe with the final table
    length = len(profile.index)
    data = pd.DataFrame({'id': profile['id'], 'gender': profile['gender'],
                         'age': profile['age'], 'income': pd.Series(profile['income'], dtype='int'),
                         'bogo': [0] * length, 'disc': [0] * length, 'info': [0] * length,
                         'amount_spend': [0] * length})
    # apply data transformation
    data = data.apply(transform, axis=1, args=(transcript, {'bogo': bogo, 'disc': disc, 'info': info}))

    data.to_pickle("data/data.pkl")


if __name__ == "__main__":
    start = time.time()
    portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
    profile = pd.read_json('data/profile.json', orient='records', lines=True)
    transcript = pd.read_json('data/transcript.json', orient='records', lines=True)

    preprocess(portfolio, profile, transcript)
    print(time.time() - start)
