from data_utils import (
    load_csvs,
    turn_into_summary_df,
    add_derived_fields,
    infer_surfaces,
)
from tqdm import tqdm
import os

# TODO: Would be nice to generate the set-score line and inspect some
# TODO: Surface would be good to add, too

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
    cur_df = add_derived_fields(cur_df)
    cur_df["surface"] = infer_surfaces(cur_tourney, cur_df["year"])
    tourney_dfs[cur_tourney] = cur_df

for cur_tourney, cur_df in tqdm(tourney_dfs.items()):
    for cur_tour in ["Womens"]:
        summarised = turn_into_summary_df(cur_df, tour=cur_tour)
        summarised.to_csv(
            os.path.join(output_dir, f"{cur_tourney}_{cur_tour.lower()}.csv")
        )
