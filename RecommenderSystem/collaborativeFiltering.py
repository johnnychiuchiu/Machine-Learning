import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
import os

class collaborativeFiltering():
    def __init__(self):
        pass

    def readSongData(self, top):
        """
        Read song data from targeted url
        """
        if 'song.pkl' in os.listdir('_data/'):
            song_df = pd.read_pickle('_data/song.pkl')
        else:
            # Read userid-songid-listen_count triplets
            # This step might take time to download data from external sources
            triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'
            songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'

            song_df_1 = pd.read_table(triplets_file, header=None)
            song_df_1.columns = ['user_id', 'song_id', 'listen_count']

            # Read song  metadata
            song_df_2 = pd.read_csv(songs_metadata_file)

            # Merge the two dataframes above to create input dataframe for recommender systems
            song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")



            # Merge song title and artist_name columns to make a merged column
            song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']

            n_users = song_df.user_id.unique().shape[0]
            n_items = song_df.song_id.unique().shape[0]
            print(str(n_users) + ' users')
            print(str(n_items) + ' items')

            song_df.to_pickle('_data/song.pkl')

        # keep top_n rows of the data
        song_df = song_df.head(top)

        song_df = self.drop_freq_low(song_df)

        return(song_df)

    def drop_freq_low(self, song_df):
        freq_df = song_df.groupby(['user_id']).agg({'song_id': 'count'}).reset_index(level=['user_id'])
        below_userid = freq_df[freq_df.song_id <= 5]['user_id']
        new_song_df = song_df[~song_df.user_id.isin(below_userid)]

        return(new_song_df)

    def utilityMatrix(self, song_df):
        """
        Transform dataframe into utility matrix, return both dataframe and matrix format
        :param song_df: a dataframe that contains user_id, song_id, and listen_count
        :return: dataframe, matrix
        """
        song_reshape = song_df.pivot(index='user_id', columns='song_id', values='listen_count')
        song_reshape = song_reshape.fillna(0)
        ratings = song_reshape.as_matrix()
        return(song_reshape, ratings)

    def fast_similarity(self, ratings, kind='user', epsilon=1e-9):
        """
        Calculate the similarity of the rating matrix
        :param ratings: utility matrix
        :param kind: user-user sim or item-item sim
        :param epsilon: small number for handling dived-by-zero errors
        :return: correlation matrix
        """

        if kind == 'user':
            sim = ratings.dot(ratings.T) + epsilon
        elif kind == 'item':
            sim = ratings.T.dot(ratings) + epsilon
        norms = np.array([np.sqrt(np.diagonal(sim))])
        return (sim / norms / norms.T)

    def predict_fast_simple(self, ratings, kind='user'):
        """
        Calculate the predicted score of every song for every user.
        :param ratings: utility matrix
        :param kind: user-user sim or item-item sim
        :return: matrix contains the predicted scores
        """

        similarity = self.fast_similarity(ratings, kind)

        if kind == 'user':
            return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
        elif kind == 'item':
            return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

    def get_overall_recommend(self, ratings, song_reshape, user_prediction, top_n=10):
        """
        get the top_n predicted result of every user. Notice that the recommended item should be the song that the user
         haven't listened before.
        :param ratings: utility matrix
        :param song_reshape: utility matrix in dataframe format
        :param user_prediction: matrix with predicted score
        :param top_n: the number of recommended song
        :return: a dict contains recommended songs for every user_id
        """
        result = dict({})
        for i, row in enumerate(ratings):
            user_id = song_reshape.index[i]
            result[user_id] = {}
            zero_item_list = np.where(row == 0)[0]
            prob_list = user_prediction[i][np.where(row == 0)[0]]
            song_id_list = np.array(song_reshape.columns)[zero_item_list]
            result[user_id]['recommend'] = sorted(zip(song_id_list, prob_list), key=lambda item: item[1], reverse=True)[
                                           0:top_n]

        return (result)

    def get_user_recommend(self, user_id, overall_recommend, song_df):
        """
        Get the recommended songs for a particular user using the song information from the song_df
        :param user_id:
        :param overall_recommend:
        :return:
        """
        user_score = pd.DataFrame(overall_recommend[user_id]['recommend']).rename(columns={0: 'song_id', 1: 'score'})
        user_recommend = pd.merge(user_score,
                                  song_df[['song_id', 'title', 'release', 'artist_name', 'song']].drop_duplicates(),
                                  on='song_id', how='left')
        return (user_recommend)

    def createNewObs(self, artistName, song_reshape, index_name):
        """
        Append a new row with userId 0 that is interested in some specific artists
        :param artistName: a list of artist names
        :return: dataframe, matrix
        """
        interest = []
        for i in song_reshape.columns:
            if i in song_df[song_df.artist_name.isin(artistName)]['song_id'].unique():
                interest.append(1)
            else:
                interest.append(0)

        print(pd.Series(interest).value_counts())

        newobs = pd.DataFrame([interest],
                              columns=song_reshape.columns)
        newobs.index = [index_name]

        new_song_reshape = pd.concat([song_reshape, newobs])
        new_ratings = new_song_reshape.as_matrix()
        return (new_song_reshape, new_ratings)

class ExplicitMF:
    """
    Train a matrix factorization model using Alternating Least Squares
    to predict empty entries in a matrix

    Parameters
    ----------
    n_iters : int
        number of iterations to train the algorithm

    n_factors : int
        number of latent factors to use in matrix
        factorization model, some machine-learning libraries
        denote this as rank

    reg : float
        regularization term for item/user latent factors,
        since lambda is a keyword in python we use reg instead
    """

    def __init__(self, n_iters, n_factors, reg):
        self.reg = reg
        self.n_iters = n_iters
        self.n_factors = n_factors

    def fit(self, train):#, test
        """
        pass in training and testing at the same time to record
        model convergence, assuming both dataset is in the form
        of User x Item matrix with cells as ratings
        """
        self.n_user, self.n_item = train.shape
        self.user_factors = np.random.random((self.n_user, self.n_factors))
        self.item_factors = np.random.random((self.n_item, self.n_factors))

        # record the training and testing mse for every iteration
        # to show convergence later (usually, not worth it for production)
        # self.test_mse_record = []
        # self.train_mse_record = []
        for _ in range(self.n_iters):
            self.user_factors = self._als_step(train, self.user_factors, self.item_factors)
            self.item_factors = self._als_step(train.T, self.item_factors, self.user_factors)
            predictions = self.predict()
            print(predictions)
            # test_mse = self.compute_mse(test, predictions)
            # train_mse = self.compute_mse(train, predictions)
            # self.test_mse_record.append(test_mse)
            # self.train_mse_record.append(train_mse)

        return self

    def _als_step(self, ratings, solve_vecs, fixed_vecs):
        """
        when updating the user matrix,
        the item matrix is the fixed vector and vice versa
        """
        A = fixed_vecs.T.dot(fixed_vecs) + np.eye(self.n_factors) * self.reg
        b = ratings.dot(fixed_vecs)
        A_inv = np.linalg.inv(A)
        solve_vecs = b.dot(A_inv)
        return solve_vecs

    def predict(self):
        """predict ratings for every user and item"""
        pred = self.user_factors.dot(self.item_factors.T)
        return pred


def compute_mse(y_true, y_pred):
    """ignore zero terms prior to comparing the mse"""
    mask = np.nonzero(y_true)
    mse = mean_squared_error(y_true[mask], y_pred[mask])
    return mse



def create_train_test(ratings):
    """
    split into training and test sets,
    remove 3 ratings from each user
    and assign them to the test set
    """
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_index = np.random.choice(
            np.flatnonzero(ratings[user]), size=3, replace=False)

        train[user, test_index] = 0.0
        test[user, test_index] = ratings[user, test_index]

    # assert that training and testing set are truly disjoint
    assert np.all(train * test == 0)
    return (train, test)

if __name__=="__main__":
    cf = collaborativeFiltering()
    song_df = cf.readSongData(top=10000)
    # Get the utility matrix
    song_reshape, ratings = cf.utilityMatrix(song_df)

    # Append new rows to simulate a users who love coldplay and Eninem
    song_reshape, ratings = cf.createNewObs(['Beyoncé', 'Katy Perry', 'Alicia Keys'], song_reshape, 'GirlFan')
    song_reshape, ratings = cf.createNewObs(['Metallica', 'Guns N\' Roses', 'Linkin Park', 'Red Hot Chili Peppers'],
                                            song_reshape, 'HeavyFan')
    song_reshape, ratings = cf.createNewObs(['Daft Punk','John Mayer','Hot Chip','Coldplay','Alicia Keys'],
                                            song_reshape, 'Johnny')

    song_reshape = song_reshape.copy()

    train, test = create_train_test(ratings)

    # Calculate user-user collaborative filtering
    user_prediction = cf.predict_fast_simple(train, kind='user')
    user_overall_recommend = cf.get_overall_recommend(train, song_reshape, user_prediction, top_n=10)
    user_recommend_girl = cf.get_user_recommend('GirlFan', user_overall_recommend, song_df)
    user_recommend_heavy = cf.get_user_recommend('HeavyFan', user_overall_recommend, song_df)
    user_recommend_johnny = cf.get_user_recommend('Johnny', user_overall_recommend, song_df)
    user_mse = compute_mse(test, user_prediction)

    # Calculate item-item collaborative filtering
    item_prediction = cf.predict_fast_simple(train, kind='item')
    item_overall_recommend = cf.get_overall_recommend(train, song_reshape, item_prediction, top_n=10)
    item_recommend_girl = cf.get_user_recommend('GirlFan', item_overall_recommend, song_df)
    item_recommend_heavy = cf.get_user_recommend('HeavyFan', item_overall_recommend, song_df)
    item_recommend_johnny = cf.get_user_recommend('Johnny', item_overall_recommend, song_df)
    item_mse = compute_mse(test, item_prediction)

    # Recommend using Latent Factor Model
    als = ExplicitMF(n_iters=50, n_factors=3, reg=0.01)
    als.fit(train)

    latent_prediction = als.predict()
    latent_overall_recommend = cf.get_overall_recommend(train, song_reshape, latent_prediction, top_n=10)
    latent_recommend_girl = cf.get_user_recommend('GirlFan', latent_overall_recommend, song_df)
    latent_recommend_heavy = cf.get_user_recommend('HeavyFan', latent_overall_recommend, song_df)
    latent_recommendjohnny = cf.get_user_recommend('Johnny', latent_overall_recommend, song_df)
    latent_mse = compute_mse(test, latent_prediction)

    pass