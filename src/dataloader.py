import pandas as pd


def base_loader(split):
    # replace split by 'test_2' for FINAL submission

    df = pd.read_csv('../data/tweets_train.csv')
    df_test = pd.read_csv(f'../data/tweets_{split}.csv')

    df['words_str'] = df['words'].apply(lambda words: ' '.join(eval(words)))
    df_test['words_str'] = df_test['words'].apply(lambda words: ' '.join(eval(words)))

    return df, df_test
