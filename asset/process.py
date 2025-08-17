import polars as pl

def generate_game1_csv(path: str, file_name: str):
    df = pl.read_csv(path, skip_rows=2)
    df_ball = df.select(["Ball", "_duplicated_13"])
    df_player = df.drop(["Ball", "_duplicated_13"])
    new_names = []
    player_name = None
    for i, name in enumerate(df_player.columns):
        if name.startswith("Player"):
            player_name = name
            new_names.append(f"{name}_x")   # or any scheme you like
        elif name.startswith("_duplicated_"):
            new_names.append(f"{player_name}_y")
        elif name == "":
            new_names.append(f"{player_name}_y")
        else:
            if player_name is not None:
                new_names.append(f"{player_name}_y")
            new_names.append(name)

    df_player = df_player.rename(dict(zip(df_player.columns, new_names)))
    df_ball = df_ball.rename({"_duplicated_13": "Ball_y", "Ball": "Ball_x"})
    df = pl.concat([df_player, df_ball], how="horizontal")
    df.write_csv(f"../{file_name}")

if __name__ == "__main__":
    generate_game1_csv("raw_data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
                 "Game1_home_tracking.csv")
    generate_game1_csv("raw_data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
                 "Game1_away_tracking.csv")