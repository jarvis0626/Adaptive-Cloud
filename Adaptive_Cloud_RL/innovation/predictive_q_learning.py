from algorithms.q_learning import QLearning
from environment.cloud_environment import arrival_band, state_to_index


class PredictiveQLearning(QLearning):
    name = "Predictive Q-Learning"
    predictive = True

    def encode(self, state, info):
        return state_to_index(state + (arrival_band(info["forecast_next"]),))

