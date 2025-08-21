import math

import numpy as np
import polars as pl
from typing import List, Tuple, Literal

from src.pitchcontrol.football_enetities import Player, Team, Ball, FrameWrapper, Vector2
from src.pitchcontrol.constants import PITCH_LENGTH, PITCH_WIDTH


class MatchParser:
    def __init__(self, frame: int, home_team_path: str, away_team_path: str, event_path: str=None):
        self.df_home_team = pl.read_csv(home_team_path)
        self.df_away_team = pl.read_csv(away_team_path)

        self.home_team, self.ball = self.init_team_frame_info(self.df_home_team, frame)
        self.away_team, _ = self.init_team_frame_info(self.df_away_team, frame)

    def init_team_frame_info(self, df: pl.DataFrame, frame: int)->Tuple[Team,Ball]:
        team_index = df.columns
        team_frame_td = df.row(frame)
        team_wrapper = self.get_frame_team_ball(team_index, team_frame_td)
        team: Team = team_wrapper.team
        ball: Ball = team_wrapper.ball

        return team, ball

    def generate_frame_pitch_control(self, side: Literal["home", "away"]="home", plot:bool=False)->np.ndarray[np.float32]:
        home_pitch_value: np.float32 = self.home_team.team_pitch_value(self.ball)
        away_pitch_value: np.float32 = self.away_team.team_pitch_value(self.ball)

        pitch_control = home_pitch_value - np.float32(away_pitch_value) if side == "home" \
            else away_pitch_value - np.float32(home_pitch_value)
        pitch_control = 1 / (1 + np.exp(-pitch_control))
        if plot:
            import matplotlib.pyplot as plt

            plt.imshow(pitch_control, cmap='coolwarm', origin='lower')
            plt.colorbar(label='Heat value')
            plt.title("Football Pitch Heatmap")
            plt.xlabel("Width (m)")
            plt.ylabel("Length (m)")

            plt.show()

        return pitch_control

    @staticmethod
    def get_frame_team_ball(team_index: List[str], team_frame_td: Tuple[float]) -> FrameWrapper:
        player_list: List[Player] = []
        ball = Ball(position=Vector2(0 ,0), velocity=Vector2(0, 0))
        for i in range(3,len(team_index)):
            c_name = team_index[i]
            player, direction = c_name.split("_")[0], c_name.split("_")[1]
            if direction == "y" or c_name.endswith("v") or math.isnan(team_frame_td[i]):
                continue

            attr_index = [i]
            for j in range(i+1, len(team_index)):
                if team_index[j].split("_")[0] == player:
                    attr_index.append(j)

            pos_x = team_frame_td[attr_index[0]] * PITCH_LENGTH
            pos_y = team_frame_td[attr_index[1]] * PITCH_WIDTH
            v_x = team_frame_td[attr_index[2]] * PITCH_LENGTH
            v_y = team_frame_td[attr_index[3]] * PITCH_WIDTH

            if c_name.lower().startswith("player"):
                player = Player(Vector2(pos_x, pos_y), Vector2(v_x, v_y))
                player.player_name = c_name.split("_")[0]
                player_list.append(player)
            if c_name.lower().startswith("ball"):
                if ball.instance is None:
                    ball = Ball(Vector2(pos_x, pos_y), Vector2(v_x, v_y))
                else:
                    ball.position = Vector2(pos_x, pos_y)
                    ball.velocity = Vector2(v_x, v_y)
        assert player_list.__len__() <= 11, "More than 11 players are on the pitch..."

        return FrameWrapper(Team(player_list), ball)