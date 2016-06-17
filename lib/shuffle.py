import random

def shuffle(df):
    # Source: http://stackoverflow.com/a/25319311/1501575
    index = list(df.index)
    random.shuffle(index)
    df = df.ix[index]
    df.reset_index()
    return df

