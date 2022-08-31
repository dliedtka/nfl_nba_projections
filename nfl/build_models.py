#!/usr/bin/env python3


import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression


cur_dir = os.path.dirname(os.path.abspath(__file__))


def quarterback_data():
    '''
    Find all QBs from data.pkl.
    At first, construct dataset of combine stats and number of seasons played, draft position. 
    Negative seasons played indicates a still active QB.
    '''
    # grab qb name/link, draft position, combine stats, career stats
    with open(f"{cur_dir}/data/data.pkl", "rb") as fin:
        data = pickle.load(fin)
    
    qb_data = {}
    qb_data["name"] = []
    qb_data["link"] = []
    qb_data["draft_pos"] = []
    qb_data["seasons"] = []
    qb_data["combine"] = []
    # career stats (passing, rushing)
    
    for idx in range(len(data["tracked_players"])):
        # 0 index for position in combine stats
        if data["combine_stats"][idx][0].lower() == "qb":
            
            # 3 index for passing table
            passing_table = data["career_stats"][idx][3]
            if passing_table is not None:
                seasons = len(passing_table)
                # using negative for still active and under 6 seasons (want to skew to longer careers)
                if passing_table[-1][0] == 2021 and seasons < 6:
                    seasons *= -1
                qb_data["seasons"].append(seasons)
            else:
                continue

            # 0 index for name
            qb_data["name"].append(data["tracked_players"][idx][0])
            # 1 index for link
            qb_data["link"].append(data["tracked_players"][idx][1])
            qb_data["draft_pos"].append(data["draft_pick"][idx])
            # don't need position, ignoring age (less fun)
            qb_data["combine"].append(data["combine_stats"][idx][2:])

    # debugging
    #for i in range(len(qb_data["name"])):
    #    print (qb_data["name"][i], qb_data["draft_pos"][i], qb_data["seasons"][i], qb_data["combine"][i])
    
    # seasons played dataset (nan if exercise not performed)
    if not os.path.exists(f"{cur_dir}/data/qb_seasons.pkl"):
        x = []
        y = []
        for i in range(len(qb_data["name"])):
            if qb_data["seasons"][i] >= 0:
                x.append(qb_data["combine"][i])
                y.append([qb_data["seasons"][i], qb_data["draft_pos"][i]])
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        with open(f"{cur_dir}/data/qb_seasons.pkl", "wb") as fout:
            pickle.dump((x,y), fout)
    

def quarterback_predict():
    '''
    Will want to predict seasons played. (normalize data)
    Then, for each season: 
    -games started, passing attempts (per game?) passing yards (per game?), passing TDs (per game?), passing INTs (per game?)
    -rushing attempts (per game?), rushing yards (per game?), rushing TDs (per game)
    -sacks (per game?), fumbles (per game?)
    -passer rating?
    '''
    # load x,y from seasons
    with open(f"{cur_dir}/data/qb_seasons.pkl", "rb") as fin:
        x, y = pickle.load(fin)
    
    # get medians (not including nan) for each exercise 
    medians = []
    for col in range(x.shape[1]):
        vals = list(filter(lambda x: not np.isnan(x), x[:,col]))
        vals.sort()
        medians.append(vals[len(vals)//2])
    #print (medians)

    # one hot cols
    one_hot = np.isnan(x).astype(float)
    #print (one_hot)

    # replace nan with medians
    #print (x)
    for col, median in enumerate(medians):
        x[:,col][np.nonzero(one_hot[:,col])] = median
    #print (x)

    # add one hot cols to data
    x = np.hstack((x, one_hot))
    #print (x)

    # train, predict
    model = LinearRegression()
    model.fit(x, y)     


if __name__ == "__main__":
    quarterback_data()
    quarterback_predict()
