#!/usr/bin/env python3


import os
import pickle


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
    # combine stats
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

    # debugging
    total = 0
    count = 0
    for i in range(len(qb_data["name"])):
        print (qb_data["name"][i], qb_data["draft_pos"][i], qb_data["seasons"][i])
        if qb_data["seasons"][i] >= 0:
            total += qb_data["seasons"][i]
            count += 1
    print (f"Average of {total/count} seasons played by {count}/{len(qb_data['name'])} qualifying QBs.")
    

def quarterback_predict():
    '''
    Will want to predict seasons played.
    Then, for each season: 
    -games started, passing attempts (per game?) passing yards (per game?), passing TDs (per game?), passing INTs (per game?)
    -rushing attempts (per game?), rushing yards (per game?), rushing TDs (per game)
    -sacks (per game?), fumbles (per game?)
    -passer rating?
    '''
    pass


if __name__ == "__main__":
    quarterback_data()
