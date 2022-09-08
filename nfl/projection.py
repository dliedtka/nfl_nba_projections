#!/usr/bin/env python3


import os
import pickle
import numpy as np


def round(val):
    if val - int(val) >= 0.5:
        return int(val) + 1
    else:
        return int(val)


def draft_round(overall):
    return int(overall/32) + 1, overall % 32


class Projection:
    def __init__(self, position):
        self.position = position
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        if self.position == "qb":
            with open(f"{self.cur_dir}/models/qb_seasons.pkl", "rb") as fin:
                self.qb_seasons_model = pickle.load(fin)

    def predict(self, combine_stats, seasons_specified=0):
        if self.position == "qb":
            x = (combine_stats[0], combine_stats[1], combine_stats[2], combine_stats[3], 
                combine_stats[4], combine_stats[5], combine_stats[6], combine_stats[7],
                float(combine_stats[2] is not None), float(combine_stats[3] is not None),
                float(combine_stats[4] is not None), float(combine_stats[5] is not None),
                float(combine_stats[6] is not None), float(combine_stats[7] is not None))
            preds = self.qb_seasons_model.predict(np.array(x).reshape(1,-1))[0]
            seasons = round(preds[0])
            draft_overall = round(preds[1])
            draft_pos = draft_round(draft_overall)
            print (f"You are projected to be drafted at Round {draft_pos[0]}, Pick {draft_pos[1]} ({draft_overall} overall), and you are projected to play {seasons} seasons.")


def prompt_combine_stats(medians=False):
    # Ht, Wt, 40yd, Bench, Broad Jump, Shuttle, 3Cone, Vertical
    # medians [75.0, 225.0, 4.81, 22.0, 111.0, 4.28, 7.11, 31.5]
    if medians:
        return (75.0, 225.0, 4.81, 22.0, 111.0, 4.28, 7.11, 31.5)
    height = float(input("Height? "))
    weight = float(input("Weight? "))
    forty = float(input("40 yd dash? "))
    bench = float(input("Bench? "))
    broadjump = float(input("Broad jump? "))
    shuttle = float(input("Shuttle? "))
    threecone = float(input("3 Cone? "))
    vertical = float(input("Vertical? "))
    return (height, weight, forty, bench, broadjump, shuttle, threecone, vertical)


if __name__ == "__main__":
    myproj = Projection("qb")
    combine_stats = prompt_combine_stats(medians=True)
    myproj.predict(combine_stats)
