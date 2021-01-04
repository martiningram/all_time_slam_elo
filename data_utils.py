import pandas as pd
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def base_name_from_path(path):

    return os.path.split(os.path.splitext(path)[0])[1]


def load_csvs(base_path):

    csvs = glob(os.path.join(base_path, "*.csv"))
    lookup = {base_name_from_path(x): pd.read_csv(x) for x in csvs}
    return lookup


def add_derived_fields(tourney_df, drop_unknown_rounds=True):

    default = "Mens" if tourney_df["links"].str.contains("Wimbledon").any() else None

    tourney_df["tour"] = np.select(
        [
            tourney_df["links"].str.contains("Women"),
            tourney_df["links"].str.contains("Men"),
        ],
        ["Womens", "Mens"],
        default=default,
    )

    tourney_df["year"] = (
        tourney_df["links"]
        .str.split("/")
        .str.get(-1)
        .str.split("_")
        .str.get(0)
        .astype(int)
    )

    round_ordering = {
        "first round": 1,
        "second round": 2,
        "third round": 3,
        "fourth round": 4,
        "quarterfinals": 5,
        "semifinals": 6,
        "final": 7,
    }

    assert drop_unknown_rounds, "Unknown rounds must be dropped for now."

    tourney_df = tourney_df[tourney_df["round"].isin(round_ordering.keys())]
    tourney_df["round_number"] = tourney_df["round"].replace(round_ordering).astype(int)
    tourney_df = tourney_df.sort_values(["year", "round_number"])

    return tourney_df


def plot_completeness(tourney_df, tour):

    assert tour in ["Mens", "Womens"]

    subset = tourney_df[tourney_df["tour"] == tour]
    unduped = subset.drop_duplicates(subset=["match", "year", "round"])
    years = unduped["year"].value_counts()
    to_plot = years.sort_index()

    return to_plot


def summarise(match_df):

    # TODO: This function needs more testing!

    def make_nan_row():

        return {
            x: np.nan
            for x in [
                "winner",
                "loser",
                "sets_winner",
                "sets_loser",
                "games_winner",
                "games_loser",
                "surface",
            ]
        }

    games_won = defaultdict(dict)

    for row in match_df.itertuples():

        games_won[row.set][row.player] = row.gameswon

    try:
        p1, p2 = np.unique(match_df["player"].values)
    except ValueError:
        return pd.Series(make_nan_row())

    sets_won = defaultdict(lambda: 0)
    total_games = defaultdict(lambda: 0)

    for cur_set in games_won.keys():

        try:
            cur_p1_games = games_won[cur_set][p1]
            cur_p2_games = games_won[cur_set][p2]

            total_games[p1] += cur_p1_games
            total_games[p2] += cur_p2_games
        except KeyError:
            return pd.Series(make_nan_row())

        sets_won[p1] += int(cur_p1_games > cur_p2_games)
        sets_won[p2] += int(cur_p1_games < cur_p2_games)

    match_winner = p1 if sets_won[p1] > sets_won[p2] else p2
    match_loser = p2 if match_winner == p1 else p1

    info = pd.Series(
        {
            "winner": match_winner,
            "loser": match_loser,
            "sets_winner": sets_won[match_winner],
            "sets_loser": sets_won[match_loser],
            "games_winner": total_games[match_winner],
            "games_loser": total_games[match_loser],
            "surface": row.surface,
        }
    )

    return info


def turn_into_summary_df(tourney_df, tour="Mens"):

    subset = tourney_df[tourney_df["tour"] == tour]

    info = (
        subset.dropna(subset=["gameswon"])
        .groupby(["match", "year", "round", "round_number"])
        .apply(summarise)
    )

    summarised = info.reset_index()

    summarised["tourney_name"] = tourney_df["tourney_name"].iloc[0]

    # TODO: Check whether this was always so.
    summarised["tourney_order"] = summarised["tourney_name"].replace(
        {"ausopen": 1, "frenchopen": 2, "wimbledon": 3, "usopen": 4}
    )

    return summarised.sort_values(["year", "round_number"])


def infer_surfaces(tourney_name, years):
    """
    Surface research:
    * US Open was on clay from 1975 to 1977
    * US Open was on grass from 1881 to 1974
    * FO has been clay since 1891
    * AO was on grass between 1905 and 1987
    (see https://en.wikipedia.org/wiki/Tennis_court)
    """

    if tourney_name == "ausopen":
        surface_names = np.select([years < 1987, years >= 1988], ["grass", "hard"])
    elif tourney_name == "wimbledon":
        surface_names = np.repeat("grass", years.shape[0])
    elif tourney_name == "frenchopen":
        surface_names = np.repeat("clay", years.shape[0])
    else:
        assert tourney_name == "usopen"
        surface_names = np.select(
            [years <= 1974, (years >= 1975) & (years <= 1977), (years >= 1978)],
            ["grass", "clay", "hard"],
        )

    return surface_names
