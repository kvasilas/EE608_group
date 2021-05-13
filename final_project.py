import urllib.request as urllib
import json
import pandas as pd
import re
from itertools import permutations

from pulp import *


def get_float(l, key):
    for d in l:
        try:
            return float(d.get(key))
        except:
            pass
    #rtn is a float
    return 0.0


def summary(prob):
    div = '---------------------------------------\n'
    print("Team Roster :\n")
    score = str(prob.objective)
    constraints = [str(const) for const in prob.constraints.values()]
    for v in prob.variables():
        score = score.replace(v.name, str(v.varValue))
        constraints = [const.replace(v.name, str(v.varValue)) for const in constraints]
        if v.varValue != 0:
            print(v.name, "=", v.varValue)
    print(div)
    print("Team Cost:")
    for constraint in constraints:
        constraint_pretty = " + ".join(re.findall("[0-9\.]*\*1.0", constraint))
        if constraint_pretty != "":
            print("{} = {}".format(constraint_pretty, eval(constraint_pretty)))
    print(div)
    print("Score:")
    score_pretty = " + ".join(re.findall("[0-9\.]+\*1.0", score))
    print("{} = {}".format(score_pretty, eval(score)))


def eval_players(players):
    return sum([current[current.displayName == player].iloc[0].points for player in players])


def greedy(val):
    remaining = SALARY_CAP
    positions = current.position.unique()
    best_players = []
    best_so_far = -float("inf")
    for comb_position in permutations(positions):
        players = []
        for pos in comb_position:
            for _ in range(pos_num_available[pos]):
                available = current[(~current.displayName.isin(players)) & 
                                 (current.position == pos) & 
                                 (current.salary <= remaining)]
                if available.size > 0:
                    best = available.sort_values(val,ascending=False).iloc[0]
                    players.append(best.displayName)
                    remaining -= best.salary
        cur_eval = eval_players(players)
        if cur_eval > best_so_far:
            best_players = players
            best_so_far = cur_eval
    return best_players



#Read in Json from draftkings
DATA_LOC = "https://api.draftkings.com/draftgroups/v1/draftgroups/21434/draftables?format=json"
response = urllib.urlopen(DATA_LOC, timeout=1)
data = json.loads(response.read())
current = pd.DataFrame.from_dict(data["draftables"])

# Remove any playersnt playing that week
current = current[current.status == "None"]
current.head()


points = [get_float(x, "value") for x in current.draftStatAttributes]
current["points"] = points

availables = current[["position", "displayName", "salary", "points"]].groupby(["position", "displayName", "salary", "points"]).agg("count")
availables = availables.reset_index()

availables[availables.position=="QB"].head(15)

salaries = {}
points = {}
for pos in availables.position.unique():
    available_pos = availables[availables.position == pos]
    salary = list(available_pos[["displayName","salary"]].set_index("displayName").to_dict().values())[0]
    point = list(available_pos[["displayName","points"]].set_index("displayName").to_dict().values())[0]
    salaries[pos] = salary
    points[pos] = point

#positions avaliable
pos_num_available = {
    "QB": 1,
    "RB": 3,
    "WR": 4,
    "TE": 2,
    "FLEX": 1,
    "DST": 1
}
#Possibilities for a flex
pos_flex = {
    "QB": 0,
    "RB": 1,
    "WR": 1,
    "TE": 1,
    "FLEX": 0,
    "DST": 0
}
pos_flex_available = 5

salaries["DST"]

SALARY_CAP = 50000 
_vars = {k: LpVariable.dict(k, v, cat="Binary") for k, v in points.items()}

prob = LpProblem("Fantasy", LpMaximize)
rewards = []
costs = []
position_constraints = []

# Setting up the reward
for k, v in _vars.items():
    costs += lpSum([salaries[k][i] * _vars[k][i] for i in v])
    rewards += lpSum([points[k][i] * _vars[k][i] for i in v])
    prob += lpSum([_vars[k][i] for i in v]) <= pos_num_available[k]
    prob += lpSum([pos_flex[k] * _vars[k][i] for i in v]) <= pos_flex_available
    
prob += lpSum(rewards)
prob += lpSum(costs) <= SALARY_CAP
prob.solve()

summary(prob)

greedy_points = greedy("points")
print("\nProjected Points: \n",eval_players(greedy_points))

# Calculate points per dk$ spent
points_per_dollar = current.points / current.salary
current["points_per_dollar"] = points_per_dollar
#Calculeate overall
greedy_points = greedy("points_per_dollar")
print("Final Player List: \n", greedy_points)

eval_players(greedy_points)

