import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


raw_filename = 'Full_Data.csv'


def text_preprocess(raw_data):
    news_df = raw_data.iloc[:, :-1].copy() # extract the news titles
    
    # Remove the punctuations
    news_df = news_df.replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True)
    # Convert the titles to lower cases
    news_df = news_df.applymap(str.lower)
    # Combine all the titles for each date to form a series of contents
    contents = [' '.join(list(news_df.iloc[i, :])) for i in range(len(news_df))]
    # Replace multiple spaces with single space
    contents = [' '.join(content.split()) for content in contents]
    contents = pd.DataFrame(contents, index=raw_data.index, columns=['Content'])
    
    # Incorporate the news_df to the original dataframe
    new_data = contents.copy()
    new_data['Label'] = raw_data.iloc[:, -1]
    
    return new_data


def split_train_and_test(whole_df, train_pct=0.8):
    train_size = int(len(whole_df) * train_pct)
    
    train_df = whole_df.iloc[:train_size, :]
    test_df = whole_df.iloc[train_size:, :]
    
    return train_df, test_df


if __name__ == '__main__':
    #raw_df = pd.read_csv(raw_filename, index_col=0)
    new_df = text_preprocess(raw_df)
    
    #train_df, test_df = split_train_and_test(df)
