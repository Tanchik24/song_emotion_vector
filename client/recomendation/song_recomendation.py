import pandas as pd
from client.recomendation.recomendation_utils import genre_preprocess, get_sentiment_analisys, normalize
from sklearn.metrics.pairwise import cosine_similarity


def get_recommendation(ids):
    df = pd.read_csv('/Users/tanchik/music_project/vk_audio_downloader/songs_features.csv')
    genre_df = genre_preprocess(df)
    df = get_sentiment_analisys(df)
    df.drop(columns=['genres'], inplace=True)
    normalized_float_df = normalize(df)
    features_df = pd.concat([df[['id']], genre_df, normalized_float_df], axis=1)
    vector = features_df[features_df['id'].isin(ids)].drop(columns=['id']).mean(axis=0)
    all_songs = features_df[~features_df['id'].isin(ids)].drop(columns=['id'])
    try:
        res = cosine_similarity(all_songs.values, vector.values.reshape(1, -1))
    except ValueError:
        return
    recommendation = list(pd.DataFrame(res, index=features_df[~features_df['id'].isin(ids)]['id'])[0].nlargest(10).index)
    return recommendation