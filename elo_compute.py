import ast
import numpy as np
import pandas as pd
from scipy import stats

def elo_expected(R_A, R_B):
    return 1 / (1 + 10**((R_B - R_A) / 400))

def elo_update(R, K, S, E):
    return R + K * (S - E)


def rank_systems(rankings, K=32, base_elo=1500):
    # Initialize Elo ratings
    elo_ratings = {i: base_elo for i in range(0, 16)}  # Adjusted for 5 systems

    for rank in rankings:
        for i in range(len(rank) - 1):
            A, B = rank[i], rank[i+1]
            
            E_A = elo_expected(elo_ratings[A], elo_ratings[B])
            E_B = 1 - E_A

            # Adjusting ratings based on results
            elo_ratings[A] = elo_update(elo_ratings[A], K, 1, E_A)  # A is ranked higher, so S_A = 1
            elo_ratings[B] = elo_update(elo_ratings[B], K, 0, E_B)  # B is ranked lower, so S_B = 0

    # Map the Elo scores to the range [1, 5]
    elo_ratings = map_elo_to_range(elo_ratings)

    return elo_ratings

def maximum_elo_after_n_games(base_elo=1500, K=32, n=3):
    elo = base_elo
    for _ in range(n):
        E_A = elo_expected(elo, base_elo)
        elo = elo_update(elo, K, 1, E_A)  # always winning
    return elo

def minimum_elo_after_n_games(base_elo=1500, K=32, n=3):
    elo = base_elo
    for _ in range(n):
        E_A = elo_expected(elo, base_elo)
        elo = elo_update(elo, K, 0, E_A)  # always losing
    return elo

data = pd.read_csv('rankings.csv', delimiter='\t', error_bad_lines=False)
rankings = [ast.literal_eval(r) for r in data["rankings"].to_list()]
relevances = [ast.literal_eval(r) for r in data["relevance"].to_list()]
relevances = np.array(relevances).mean(axis=0)
max_elo = maximum_elo_after_n_games(n=len(rankings))
min_elo = minimum_elo_after_n_games(n=len(rankings))

def map_elo_to_range(elo_ratings, new_min=0, new_max=5, old_min=min_elo, old_max=max_elo):
    for system, rating in elo_ratings.items():
        elo_ratings[system] = new_min + (rating - old_min) * (new_max - new_min) / (old_max - old_min)
    return elo_ratings

print(relevances)
elos_dict = rank_systems(rankings)
final_elos = []
for i in range(len(elos_dict.keys())):
    final_elos.append(elos_dict[i])
print(final_elos)
res = stats.pearsonr(final_elos, relevances)
print(res)
# System-level : r=0.75 vs (0.82 QAEval)
print(rankings)
score_per_summaries = []
for l in rankings:
    if 8 not in l:
        l.append(8)
    if 15 not in l:
        l.append(15)
    local_l = []
    for system_number in range(16):
        local_l.append(l.index(system_number))
    score_per_summaries.append(list(np.interp(local_l, [0,15], [5,0])))

relevances = [ast.literal_eval(r) for r in data["relevance"].to_list()]
pearson = []
for scores_sum, relevances in zip(score_per_summaries, relevances):
    res = stats.pearsonr(final_elos, relevances)
    pearson.append(res[0])
pearson = np.array(pearson).mean()
print(pearson) # 0.36 vs (0.30 QAEval)