from dataclasses import dataclass
from typing import List
import numpy as np
import networkx as nx

from src.pitchcontrol.constants import PITCH_LENGTH, PITCH_WIDTH, MAX_SPEED, SCALE_X, SCALE_Y


class Vector2:
    def __init__(self, x: float, y:float):
        self.x = x
        self.y = y

    def __add__(self, other):
        if isinstance(other, (int, float, np.floating)):
            return Vector2(self.x + other, self.y + other)
        elif isinstance(other, Vector2):
            return Vector2(self.x + other.x, self.y + other.y)
        else:
            return NotImplemented
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        if isinstance(other, (int, float, np.floating)):
            return Vector2(self.x * other, self.y * other)
        elif isinstance(other, Vector2):
            return Vector2(self.x * other.x, self.y * other.y)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    @property
    def angle(self) -> np.float32:
        return np.atan2(self.y, self.x, dtype=np.float32)

    @property
    def vec_len(self) -> np.float32:
        return np.sqrt(self.x * self.x + self.y * self.y, dtype=np.float32)

    def euclidean_distance(self, other) -> np.float32:
        return np.sqrt((self.x - other.x).__pow__(2) + (self.y - other.y).__pow__(2), dtype=np.float32)

    def cosine_distance(self, other) -> np.float32:
        if self.vec_len == 0 or other.vec_len == 0:
            raise ValueError("Vectors must have none-zero length")
        return (self.x * other.x + self.y * other.y) / (self.vec_len * other.vec_len)

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def __repr__(self):
        return f"Vector2({self.x}, {self.y})"


class Entity:
    def __init__(self):
        self.position: Vector2
        self.velocity: Vector2


class Ball(Entity):
    def __init__(self, position: Vector2, velocity: Vector2):
        super().__init__()
        self.position = position
        self.velocity = velocity

    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return f"Ball is at {self.position}, with velocity {self.velocity}."

    @property
    def instance(self):
        return self._instance


class Player(Entity):
    def __init__(self, position: Vector2, velocity: Vector2, max_velocity:np.float32=MAX_SPEED, player_name: str=None):
        super().__init__()
        self.player_name = player_name
        self.position = position
        self.velocity = velocity

        self.max_velocity = max_velocity

    def __repr__(self):
        return f"{self.player_name} is at {self.position.x, self.position.y}, with velocity {self.velocity.x, self.velocity.y}."

    def distance_to_ball(self, ball: Ball) -> np.float32:
        return self.position.euclidean_distance(ball.position)

    def area_influence_coordinate(self, coordinate: Vector2, ball: Ball) -> np.float32:
        theta = self.velocity.angle
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]]).reshape(2, 2)
        scale = self.velocity.vec_len.__pow__(2) / np.float32(self.max_velocity).__pow__(2)
        influence_radius = self.influence_radius(self.distance_to_ball(ball))
        sx = influence_radius + np.multiply(scale, influence_radius)
        sy = influence_radius - np.multiply(scale, influence_radius)
        S = np.array([[sx * SCALE_X,   0],
                      [0,  sy * SCALE_Y]]).reshape(2, 2)
        COV = np.linalg.multi_dot([R, S, S, np.linalg.inv(R)])

        mu = (self.position + self.velocity.vec_len * np.float32(0.5)).to_numpy().reshape(2, 1)
        p = coordinate.to_numpy().reshape(2, 1)
        # const = 1 / np.sqrt(np.linalg.det(COV) * (2 * np.pi).__pow__(2))

        return np.exp(-0.5 * np.linalg.multi_dot([(p - mu).T, np.linalg.inv(COV), (p - mu)])).flatten().squeeze()

    def area_influence_surface(self, ball: Ball, pitch_array: np.ndarray[np.float32]=None) -> np.ndarray[np.float32]:
        if pitch_array is None:
            pitch_array = self.generate_pitch_array()
        theta = self.velocity.angle
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]]).reshape(2, 2)
        scale = self.velocity.vec_len.__pow__(2) / np.float32(self.max_velocity).__pow__(2)
        influence_radius = self.influence_radius(self.distance_to_ball(ball))
        sx = influence_radius + np.multiply(scale, influence_radius)
        sy = influence_radius - np.multiply(scale, influence_radius)
        S = np.array([[sx * SCALE_X,  0],
                      [0, sy * SCALE_Y]]).reshape(2, 2)
        COV = np.linalg.multi_dot([R, S, S, np.linalg.inv(R)])
        mu = (self.position + self.velocity.vec_len * np.float32(0.5)).to_numpy().reshape(2, 1)

        pitch_array = np.expand_dims(pitch_array, axis=-1)
        A = pitch_array - mu
        A_T = A.transpose(0, 1, 3, 2)
        pitch_control = np.exp(-0.5 * np.einsum("ijkl, lm, ijmn -> ijkn", A_T, np.linalg.inv(COV), A).squeeze())

        return pitch_control

    @staticmethod
    def influence_radius(ball_distance: np.float32) -> np.float32:
        # generated by ChatGPT from the fig in the paper Appendix...
        if ball_distance <= 18.5:
            return  0.00182602771*ball_distance**3 - 0.0256506927*ball_distance**2 + 0.173904156*ball_distance**1 + 4.0
        else:
            return np.float32(20)

    @staticmethod
    def generate_pitch_array():
        x = np.arange(PITCH_LENGTH)[:, None]      # shape (110, 1)
        y = np.arange(PITCH_WIDTH)[None, :]     # shape (1, 56)
        pitch_array = np.dstack(np.meshgrid(x.flatten(), y.flatten(), indexing='ij')).reshape(PITCH_LENGTH, PITCH_WIDTH, 2)
        return pitch_array


class Team:
    def __init__(self, players: List[Player]):
        self.players = players
        self.G = nx.Graph()
        self.generate_team_graph()

    def team_pitch_value(self, ball: Ball) -> np.float32:
        pitch_value = np.zeros(shape=(PITCH_LENGTH, PITCH_WIDTH), dtype=np.float32)
        for player in self.players:
            pitch_value += player.area_influence_surface(ball)

        return pitch_value

    def generate_team_graph(self):
        self.G.add_nodes_from([
            (player.player_name, {"position": player.position,
                                  "velocity": player.velocity}) for player in self.players])

        for i in range(self.players.__len__()):
            current_player = self.players[i]
            for j in range(i+1, self.players.__len__()):
                pair_player = self.players[j]
                edge_attrs: GraphEdgeAttrWrapper = self.generate_edge_attrs(current_player, pair_player)
                self.G.add_edge(current_player.player_name, pair_player.player_name, distance=edge_attrs.distance)

    @staticmethod
    def generate_edge_attrs(player: Player, pair_player: Player):
        distance = player.position.euclidean_distance(pair_player.position)

        return GraphEdgeAttrWrapper(distance=distance)


@dataclass
class FrameWrapper:
    team: Team
    ball: Ball


@dataclass
class GraphEdgeAttrWrapper:
    distance: np.float32


