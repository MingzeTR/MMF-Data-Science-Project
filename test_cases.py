import pandas as pd
from math import sqrt


def recommendation_system(inputitem_df, ratings_df, userSubsetCate_df):
    """
    A recommendation system function to generate the category with the highest score.

    :param inputitem_df:
    :param ratings_df:
    :param userSubsetCate_df:
    :return: recommendation_df:
    """

    userSubsetCateGroup_df = userSubsetCate_df.groupby(['user_id'])
    userSubsetCateGroupSample_df = sorted(userSubsetCateGroup_df, key=lambda x: len(x[1]), reverse=True)

    # calculate the avg rating for input items
    inputitem_avg = inputitem_df.groupby(['cate_name']).mean()
    inputitem_avg.reset_index(level=0, inplace=True)

    # Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
    pearsonCorrelationDict = {}

    # For every user group in our subset
    for name, group in userSubsetCateGroupSample_df:
        group = group.sort_values(by='cate_name')
        inputitem_avg = inputitem_avg.sort_values(by='cate_name')
        # Get the N for the formula
        nRatings = len(group)
        # Get the review scores for the items that they both have in common
        temp_df = inputitem_avg[inputitem_avg['cate_name'].isin(group['cate_name'].tolist())]
        # And then store them in a temporary buffer variable in a list format to facilitate future calculations
        tempRatingList = temp_df['rating'].tolist()
        # Let's also put the current user group reviews in a list format
        tempGroupList = group['rating'].tolist()
        # Now let's calculate the pearson correlation between two users, so called, x and y
        Sxx = sum([i ** 2 for i in tempRatingList]) - pow(sum(tempRatingList), 2) / float(nRatings)
        Syy = sum([i ** 2 for i in tempGroupList]) - pow(sum(tempGroupList), 2) / float(nRatings)
        Sxy = sum(i * j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList) * sum(
            tempGroupList) / float(nRatings)

        # If the denominator is different than zero, then divide, else, 0 correlation.
        if Sxx != 0 and Syy != 0:
            pearsonCorrelationDict[name] = Sxy / sqrt(Sxx * Syy)
        else:
            pearsonCorrelationDict[name] = 0

    # Transform the pearsonCorrelationDict into a panda dataframe
    pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
    pearsonDF['user_id'] = pearsonDF.index
    pearsonDF.index = range(len(pearsonDF))
    pearsonDF.columns = ['similarityIndex', 'user_id']

    # Obtain the top X similar users to input user (i.e.Pikachui)
    # set X = 50 (TBD)
    topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]

    # The next steps are sorting the items with weighted average of the
    # rating  using perason correlation as weight.
    topUsersRating = topUsers.merge(ratings_df, left_on='user_id', right_on='user_id', how='inner')

    # Multiplies the similarity by the user's ratings
    topUsersRating['weightedRating'] = topUsersRating['similarityIndex'] * topUsersRating['rating']

    # Applies a sum to the topUsers after grouping it up by userId
    tempTopUsersRating = topUsersRating.groupby('cate_name').sum()[['similarityIndex', 'weightedRating']]
    tempTopUsersRating.columns = ['sum_similarityIndex', 'sum_weightedRating']

    # Creates an empty dataframe
    recommendation_df = pd.DataFrame()
    
    # Now we take the weighted average
    recommendation_df['weighted average recommendation score'] = \
        tempTopUsersRating['sum_weightedRating'] / tempTopUsersRating['sum_similarityIndex']
    recommendation_df['cate_name'] = tempTopUsersRating.index
    recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)

    return recommendation_df


def create_userSubsetCate_df(user_id, rating_df, inputItems_df):
    """
    Function: create_userSubsetCate_df
    Input:
          user_id: (String) - id of selected Pikaqiu
          rating_df: (DataFrame) - the dataframe of overall item-base user rating records
          inputItems_df: (DataFrame) - the dataframe of all items Pikachu has rated and the corresponding ratings
                                     Output of create_inputItems_df

    Output:
          userSubsetCate_df: (DataFrame) - the dataframe of other users with common categories of item in inputItems_df
                                           and their corresponding mean rating of category-based
                                           !!! DOES NOT INCLUDE RECORDS OF PIKAQIU !!!
                             Column name: user_id(primary key for user),
                                          cate_name(the category the user has rated before)
                                          rating (average rating of each category),
    """
    # Find the categories that Pikachu has rated, and calculate the rating of each category using average
    inputCates_df = inputItems_df.groupby(['cate_name']).mean().reset_index()

    # Find all rating records of user who have rated items in same category which Pikachu has rated before (in inputCates_df)
    userSubset_df = rating_df[rating_df['cate_name'].isin(inputCates_df['cate_name'].tolist())]

    # Group all rating records by user_id and cate_name, get each user's average rating for each category
    userSubsetCate_df = userSubset_df.groupby(['user_id', 'cate_name']).mean().reset_index()

    # Drop the selected Pikaqiu from the DataFrame
    userSubsetCate_df.drop(userSubsetCate_df[userSubsetCate_df['user_id'] == user_id].index, inplace=True)

    return userSubsetCate_df




def create_inputItems_df(user_id, rating_df):
    """
    Function: create_inputItems_df
    Input:
      user_id: (String) - id of selected Pikaqiu
      rating_df: (DataFrame) - the dataframe of overall item-base user rating records

    Output:
      inputItems_df: (DataFrame) - include all items Pikachu has rated and the corresponding ratings
                     Column name: item_id(primary key for item),
                                  rating,
                                  datetime(from timestamp, converted to YYYY-MM-DD format),
                                  cate_name(the category the item belongs to)
    """
    pikachu_df = rating_df.loc[rating_df['user_id'] == user_id]
    inputItems_df = pikachu_df.drop(['user_id'], 1)

    return inputItems_df


if __name__ == '__main__':
    # # Storing the input item information into a pandas dataframe
    # inputitem_df = pd.read_csv('inputItems.csv')
    # # Storing the user information into a pandas dataframe
    ratings_df = pd.read_csv('ratings_2018.csv')
    # # Storing the mean rating of other users with common category
    # userSubsetCate_df = pd.read_csv('userSubsetCate.csv')

    # filename = "Patio_Lawn_and_Garden.csv"
    # ratings_df = pd.read_csv(filename, names=("item_id", "user_id", "rating", "timestamp"))
    # ratings_df['datetime'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
    # ratings_df.drop(['timestamp'], 1, inplace=True)
    # itemCate_df = pd.read_csv('itemCate_2018.csv', usecols=["item_id", "cate_name"])
    # ratings_df = pd.merge(ratings_df, itemCate_df, how='left', on='item_id')
    # ratings_df.to_csv('ratings_2018.csv')

    # for the test, we will use 5% of the dataset grouped by user_id. Within a user's purchase history, we save the last
    # 3 purchase records as the comparing group, and the rest purchases as inputs into the model. Calculate the
    # percentage of the output that is in the comparing group

    # we know that there are 111,233 user_id in the group, we are using 100 ids as test cases.
    test_group = ratings_df[ratings_df['user_id'].groupby(ratings_df['user_id']).transform('size') >= 10]

    success_rate = 0
    test_counts = 0
    for userid, group in test_group.groupby('user_id'):
        inputitem_df = create_inputItems_df(userid, ratings_df)
        compare_group = inputitem_df.tail(2)
        inputitem_df = inputitem_df.drop(group.tail(2).index)
        userSubsetCate_df = create_userSubsetCate_df(userid, ratings_df, inputitem_df)
        recommend_outcome = recommendation_system(inputitem_df, ratings_df, userSubsetCate_df)

        if True in \
                compare_group['cate_name'].isin(
                    recommend_outcome[recommend_outcome['weighted average recommendation score'] >= 4]['cate_name']
                ).to_list():
            success_rate += 1

        test_counts += 1
        if test_counts == 100:
            break

        # save some memory
        del inputitem_df, compare_group, userSubsetCate_df, recommend_outcome
        print(success_rate, test_counts)
    print(success_rate / 100)
