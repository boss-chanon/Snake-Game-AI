import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.activations import relu, softmax, tanh

from ENV import ENV

tf.compat.v1.disable_eager_execution()

class Agent:
    def __init__(self, env, param):
        self.state_space = env.state_space
        self.action_space = env.action_space

        self.learning_rate = param["learning_rate"]
        self.gamma = param["gamma"]

        self.model = self.built_model()

    def built_model(self):
        X_input = Input(self.state_space)

        X = Dense(128, activation=tanh)(X_input)
        X = Dense(128, activation=tanh)(X)
        X = Dense(128, activation=tanh)(X)
        output = Dense(self.action_space, activation=softmax)(X)

        model = Model(inputs=X_input, outputs=output)
        model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def act(self, state):
        action_prob = self.model.predict(state)
        return np.random.choice(a=4, p=action_prob.ravel())

    def load(self, path):
        self.model = load_model(path)

    def save(self, path):
        self.model.save(path)

    def discount_reward(self, reward):
        discounted_episode_rewards = np.zeros_like(reward)
        cumulative = 0.0
        for i in reversed(range(len(reward))):
            cumulative = cumulative * self.gamma + reward[i]
            discounted_episode_rewards[i] = cumulative

        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)
        discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

        if len(reward) <= 1:
            discounted_episode_rewards = reward

        return discounted_episode_rewards

    def train(self, state_list, reward_list, action_list):
        state_batch = np.vstack(state_list)
        action_batch = np.array(action_list)
        reward_batch = np.array(reward_list)

        discounted_epr = self.discount_reward(reward_batch)

        loss = self.model.fit(state_batch, action_batch, sample_weight=discounted_epr, epochs=1, verbose=0)

        return np.sum(loss.history['loss'])

def train_Agent(episode, env, param, resume=True):
    sum_reward = []
    running_reward = 0
    running_reward_max = 100
    agent = Agent(env, param)
    model_path = "model/PolicyGradients/model.h5"
    scores = []

    if resume:
        agent.model = load_model(model_path)

    for ep in range(episode):
        if not env.running:
            break
        state_list, reward_list, action_list = [], [], []
        state = env.reset()
        state = np.reshape(state, (1, env.state_space))
        score = 0
        while True:
            action = agent.act(state)

            action_onehot = np.zeros([env.action_space])
            action_onehot[action] = 1

            action_list.append(action_onehot)
            state_list.append(state)

            next_state, reward, done = env.game_run(action)
            score += reward
            next_state = np.reshape(next_state, (1, env.state_space))
            state = next_state

            reward_list.append(reward)

            env.show()

            if done:
                scores.append(score)
                loss = agent.train(state_list, reward_list, action_list)
                running_reward = sum(scores[-50:]) / len(scores[-50:])
                break
        print("end state: " + str(state))
        print("ep: " + str(ep+1) + " reward: " + str(score) + " runing reward: " + str(running_reward) + " Loss: " + str(loss))
        sum_reward.append(running_reward)

        if running_reward > running_reward_max:
            running_reward_max = running_reward
            agent.save(model_path)
            
    return sum_reward

if __name__ == "__main__":
    param = dict()
    param["learning_rate"] = 0.00025
    param["gamma"] = 0.95

    ep = 10

    env = ENV(AI_active=True , limit_speed=False, render=False)
    reward= train_Agent(ep, env, param, resume=True)

    plt.plot(range(len(reward)), reward)
    plt.xlabel("EP")
    plt.ylabel("Reward")
    plt.title("REWARD")
    plt.show()