from football_enetities import *

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

if __name__ == '__main__':
    test_pitch_control()