from data_utils import load_csvs, turn_into_summary_df, add_derived_fields
from tqdm import tqdm
import os


wikidraws_dir = "./wikidraws/csv/"
output_dir = "./output/"
os.makedirs(output_dir, exist_ok=True)

lookup = load_csvs(wikidraws_dir)

tourney_dfs = {
    x: y
    for x, y in lookup.items()
    if x in ["wimbledon", "frenchopen", "usopen", "ausopen"]
}

for cur_tourney, cur_df in tourney_dfs.items():
    cur_df["tourney_name"] = cur_tourney
    tourney_dfs[cur_tourney] = add_derived_fields(cur_df)

for cur_tourney, cur_df in tqdm(tourney_dfs.items()):
    # TODO: Find out what went wrong on Women's tour
    for cur_tour in ["Mens"]:
        summarised = turn_into_summary_df(cur_df, tour=cur_tour)
        summarised.to_csv(os.path.join(output_dir, f"{cur_tourney}_{cur_tour}.csv"))
