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
        self.train_freq = param["train_freq"]
        self.max_memory = param["max_memory"]
        self.batch_size = param["batch_size"]

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

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def train(self, step):
        if self.batch_size > len(self.memory) or step % self.train_freq != 0:
            return 0, 0
            
        minibatch = random.sample(self.memory, self.batch_size)
        state_batch = np.array([i[0] for i in minibatch])
        action_batch = np.array([i[1] for i in minibatch])
        reward_batch = np.array([i[2] for i in minibatch])
        next_state_batch = np.array([i[3] for i in minibatch])
        done_batch = np.array([i[4] for i in minibatch])

        state_batch = np.squeeze(state_batch)
        next_state_batch = np.squeeze(next_state_batch)

        q_values_next = self.critic.predict(next_state_batch)
        q_values = self.critic.predict(state_batch)

        q_values = np.squeeze(q_values)
        q_values_next = np.squeeze(q_values_next)

        targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * self.gamma * q_values_next

        critic_loss = self.critic.update(state_batch, targets_batch)

        targets_batch -= q_values

        actor_loss = self.actor.update(state_batch, targets_batch, action_batch)

        return actor_loss, critic_loss

def train_Agent(episode, env, param, resume=True):
    sum_reward = []
    scores = []
    step = 0
    running_reward = None
    running_reward_max = 100
    agent = Agent(env, param)
    model_path = "model/AC/"

    if resume:
        agent.load(model_path)
    for ep in range(episode):
        if not env.running:
            break
        state = env.reset()
        state = np.reshape(state, (1, env.state_space))
        score = 0
        while True:
            step += 1
            action = agent.act(state)
            next_state, reward, done = env.game_run(action)
            next_state = np.reshape(next_state, (1, env.state_space))
            score += reward

            action_onehot = np.zeros([env.action_space])
            action_onehot[action] = 1

            agent.remember(state, action_onehot, reward, next_state, done)

            a_loss, c_loss = agent.train(step)

            state = next_state

            env.show()

            if done:
                scores.append(score)
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
    param["batch_size"] = 500
    param["gamma"] = 0.95
    param["train_freq"] = 1
    param["max_memory"] = 2000

    ep = 10

    env = ENV(AI_active=True , limit_speed=False, render=True)
    reward= train_Agent(ep, env, param, resume=True)

    plt.plot(range(len(reward)), reward)
    plt.xlabel("EP")
    plt.ylabel("Reward")
    plt.title("REWARD")
    plt.show()