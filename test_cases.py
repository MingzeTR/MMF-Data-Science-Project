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


if __name__ == '__main__':
    # Storing the input item information into a pandas dataframe
    inputitem_df = pd.read_csv('inputItems.csv')
    # Storing the user information into a pandas dataframe
    ratings_df = pd.read_csv('ratings.csv')
    # Storing the mean rating of other users with common category
    userSubsetCate_df = pd.read_csv('userSubsetCate.csv')

    # for the test, we will use 5% of the dataset grouped by user_id. Within a user's purchase history, we save the last
    # 3 purchase records as the comparing group, and the rest purchases as inputs into the model. Calculate the
    # percentage of the output that is in the comparing group

    # we know that there are 111,233 user_id in the group, we are using 100 ids as test cases.
    test_group = userSubsetCate_df[
        userSubsetCate_df['user_id'].groupby(userSubsetCate_df['user_id']).transform('size') >= 4
        ]

    success_rate = 0
    test_counts = 0
    for userid, group in test_group.groupby('user_id'):
        compare_group = group.tail(2)
        inputitem_df = group.drop(group.tail(2).index)
        recommend_outcome = recommendation_system(inputitem_df, ratings_df, userSubsetCate_df)

        if True in \
                compare_group['cate_name'].isin(
                    recommend_outcome[recommend_outcome['weighted average recommendation score'] >= 4]['cate_name']
                ).to_list():
            success_rate += 1

        test_counts += 1
        if test_counts == 100:
            break

        print(success_rate, test_counts)
    print(success_rate / 100)
