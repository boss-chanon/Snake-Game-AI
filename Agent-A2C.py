import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import MSE, categorical_crossentropy
from tensorflow.keras.activations import relu, tanh, softmax
from tensorflow.keras.optimizers import Adam

from ENV import ENV

tf.compat.v1.disable_eager_execution()

class Critic:
    def __init__(self, state_space, action_space, learning_rate):
        X_input = Input(state_space)

        X = Dense(128, activation=relu)(X_input)
        X = Dense(128, activation=relu)(X)
        X = Dense(128, activation=relu)(X)
        output = Dense(1, activation=None)(X)

        self.model = Model(inputs=X_input, outputs=output)
        self.model.compile(loss=MSE, optimizer=Adam(learning_rate=learning_rate))
    
    def predict(self, state):
        return self.model.predict_on_batch(state)

    def update(self, state, reward):
        loss = self.model.fit(state, reward, verbose=0, epochs=1)
        
        return np.sum(loss.history['loss'])

class Actor:
    def __init__(self, state_space, action_space, learning_rate):
        X_input = Input(state_space)

        X = Dense(128, activation=tanh)(X_input)
        X = Dense(128, activation=tanh)(X)
        X = Dense(128, activation=tanh)(X)
        output = Dense(action_space, activation=softmax)(X)

        self.model = Model(inputs=X_input, outputs=output)
        self.model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=learning_rate))

    def predict(self, state):
        return self.model.predict(state)

    def update(self, state, reward, action):
        loss = self.model.fit(state, action, sample_weight=reward, verbose=0, epochs=1)

        return np.sum(loss.history['loss'])

class Agent:
    def __init__(self, env, param):
        self.state_space = env.state_space
        self.action_space = env.action_space

        self.learning_rate = param["learning_rate"]
        self.gamma = param["gamma"]

        self.memory = []

        self.built_model()

    def save(self, path):
        self.actor.model.save(path+"actor.h5")
        self.critic.model.save(path+"critic.h5")

    def load(self, path):
        self.actor.model = load_model(path+"actor.h5")
        self.critic.model = load_model(path+"critic.h5")

    def built_model(self):
        self.actor = Actor(self.state_space, self.action_space, learning_rate=self.learning_rate)
        self.critic = Critic(self.state_space, self.action_space, learning_rate=self.learning_rate)

    def act(self, state):
        action_prob = self.actor.predict(state)
        return np.random.choice(a=4, p=action_prob.ravel())

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

        values = self.critic.predict(state_batch)
        values = np.squeeze(values)

        discounted_epr = self.discount_reward(reward_batch)

        targets_batch = discounted_epr - values

        critic_loss = self.critic.update(state_batch, discounted_epr)
        actor_loss = self.actor.update(state_batch, targets_batch, action_batch)

        return actor_loss, critic_loss

def train_Agent(episode, env, param, resume=True):
    sum_reward = []
    scores = []
    step = 0
    running_reward = None
    running_reward_max = 100
    agent = Agent(env, param)
    model_path = "model/A2C/"

    if resume:
        agent.load(model_path)
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
                a_loss, c_loss = agent.train(state_list, reward_list, action_list)
                running_reward = sum(scores[-50:]) / len(scores[-50:])
                break
        print("end state: " + str(state))
        print("ep: " + str(ep+1) + " reward: " + str(score) + " running reward: " + str(running_reward) + " actor loss: " + str(a_loss) +  " critic loss: " + str(c_loss))
        sum_reward.append(running_reward)

        if running_reward > running_reward_max:
            agent.save(model_path)
    return sum_reward

if __name__ == "__main__":
    param = dict()
    param["learning_rate"] = 0.00025
    param["gamma"] = 0.95

    ep = 10

    env = ENV(AI_active=True , limit_speed=True, render=True)
    reward= train_Agent(ep, env, param, resume=True)

    plt.plot(range(len(reward)), reward)
    plt.xlabel("EP")
    plt.ylabel("Reward")
    plt.title("REWARD")
    plt.show()