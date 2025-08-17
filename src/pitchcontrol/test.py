from typing import Tuple

from football_enetities import *
import polars as pl
import math


def test_pitch_control():
    player1 = Player(Vector2(15, 15), Vector2(4.5, 4.5))
    player2 = Player(Vector2(30, 40), Vector2(0, 0))

    player3 = Player(Vector2(13, 15), Vector2(1.5, 1))
    player4 = Player(Vector2(28, 40), Vector2(0, 0))

    ball = Ball(position=Vector2(1, 15), velocity=Vector2(0, 0))
    team1 = Team(players=[player1, player2])
    team2 = Team(players=[player3, player4])
    team1_value = team1.team_pitch_value(ball)
    team2_value = team2.team_pitch_value(ball)
    pc = team1_value - team2_value
    pc = 1 / (1 + np.exp(-pc))

    import matplotlib.pyplot as plt

    plt.imshow(pc, cmap='coolwarm', origin='lower')
    plt.colorbar(label='Heat value')
    plt.title("Football Pitch Heatmap")
    plt.xlabel("Width (m)")
    plt.ylabel("Length (m)")

    plt.plot(player1.position.y, player1.position.x, '.')
    plt.plot(player2.position.y, player2.position.x, '.')
    plt.plot(player3.position.y, player3.position.x, '.')
    plt.plot(player4.position.y, player4.position.x, '.')
    plt.plot(ball.position.y, ball.position.x, 'o', color='white')
    plt.show()


def get_team(team_index: List[str], team: List[float]) -> Tuple[Team, Ball]:
    player_list: List[Player] = []
    for i in range(3,len(team_index)):
        c_name = team_index[i]
        player, direction = c_name.split("_")[0], c_name.split("_")[1]
        if direction == "y" or c_name.endswith("v") or math.isnan(team[i]):
            continue

        attr_index = [i]
        for j in range(i+1, len(team_index)):
            if team_index[j].split("_")[0] == player:
                attr_index.append(j)
        pos_x = team[attr_index[0]] * 105
        pos_y = team[attr_index[1]] * 68
        v_x = team[attr_index[2]] * 105
        v_y = team[attr_index[3]] * 68

        if c_name.lower().startswith("player"):
            player = Player(Vector2(pos_x, pos_y), Vector2(v_x, v_y))
            player_list.append(player)
        if c_name.lower().startswith("ball"):
            ball = Ball(Vector2(pos_x, pos_y), Vector2(v_x, v_y))

    return Team(player_list), ball

def test_pitch_control2(frame=38419):
    home_team_index = pl.read_csv("../../asset/Game1_home_tracking.csv").columns
    home_team = pl.read_csv("../../asset/Game1_home_tracking.csv").row(frame)

    away_team_index = pl.read_csv("../../asset/Game1_away_tracking.csv").columns
    away_team = pl.read_csv("../../asset/Game1_away_tracking.csv").row(frame)

    home_team, ball1 = get_team(home_team_index, home_team)
    away_team, ball2 = get_team(away_team_index, away_team)

    assert ball1 is ball2, "ball is not the same entity..."

    pc = home_team.team_pitch_value(ball1) - away_team.team_pitch_value(ball2)
    pc = 1 / (1 + np.exp(-pc))

    import matplotlib.pyplot as plt

    plt.imshow(pc, cmap='coolwarm', origin='lower')
    plt.colorbar(label='Heat value')
    plt.title("Football Pitch Heatmap")
    plt.xlabel("Width (m)")
    plt.ylabel("Length (m)")

    plt.show()


if __name__ == '__main__':
    test_pitch_control2()