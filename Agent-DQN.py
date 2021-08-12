import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.losses import MSE

from ENV import ENV

tf.compat.v1.disable_eager_execution()

class Agent:
    def __init__(self, env, param, is_train = True):
        self.state_space = env.state_space
        self.action_space = env.action_space

        self.learning_rate = param["learning_rate"]
        self.batch_size = param["batch_size"]
        self.gamma = param["gamma"]
        self.train_freq = param["train_freq"]
        self.max_memory = param["max_memory"]
        self.epsilon_decay_step = param["epsilon_decay_step"]

        self.is_train = is_train

        self.epsilons = np.linspace(1, 0.01, self.epsilon_decay_step)

        self.memory = []

        self.model = self.built_model()

    def built_model(self):
        X_input = Input(self.state_space)

        X = Dense(128, activation=relu)(X_input)
        X = Dense(128, activation=relu)(X)
        X = Dense(128, activation=relu)(X)
        output = Dense(self.action_space, activation=softmax)(X)

        model = Model(inputs=X_input, outputs=output)
        model.compile(loss=MSE, optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def load(self, path):
        self.model = load_model(path)

    def save(self, path):
        self.model.save(path)

    def act(self, episode, state):
        if self.is_train:
            if self.epsilon_decay_step == 0:
                self.epsilon = 0.01
            else:
                self.epsilon = self.epsilons[min(episode, self.epsilon_decay_step-1)]
        else:
            self.epsilon = 0

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        qValues = self.model.predict(state)
        return np.argmax(qValues[0])

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) > self.max_memory:
                self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def train(self, step):
        if self.batch_size > len(self.memory) or step % self.train_freq != 0:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        state_batch = np.array([i[0] for i in minibatch])
        action_batch = np.array([i[1] for i in minibatch])
        reward_batch = np.array([i[2] for i in minibatch])
        next_state_batch = np.array([i[3] for i in minibatch])
        done_batch = np.array([i[4] for i in minibatch])

        state_batch = np.squeeze(state_batch)
        next_state_batch = np.squeeze(next_state_batch)

        q_values_next = self.model.predict_on_batch(next_state_batch)

        targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * self.gamma * np.amax(q_values_next, axis=1)

        targets_full = self.model.predict_on_batch(state_batch)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [action_batch]] = targets_batch

        loss = self.model.fit(state_batch, targets_full, epochs=1, verbose=0)

        return np.sum(loss.history['loss'])

def train_Agent(episode, env, param, resume=True, is_train=True):
    sum_reward = []
    step = 0
    running_reward = None
    running_reward_max = 100
    agent = Agent(env, param, is_train=is_train)
    scores = []
    model_path = "model/DQN/model.h5"

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
            action = agent.act(ep, state)
            next_state, reward, done = env.game_run(action)
            score += reward
            next_state = np.reshape(next_state, (1, env.state_space))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            loss = agent.train(step)

            env.show()

            if done:
                scores.append(score)
                running_reward = sum(scores[-50:]) / len(scores[-50:])
                break
        if running_reward > running_reward_max:
            running_reward_max = running_reward
            agent.save(model_path)
        print("end state: " + str(state))
        print("ep: " + str(ep+1) + " epsilon: " + str(agent.epsilon) + " reward: " + str(score) + " running reward: " + str(running_reward) + " loss: " + str(loss))
        sum_reward.append(running_reward)

    return sum_reward

if __name__ == "__main__":
    param = dict()
    param["learning_rate"] = 0.00025
    param["batch_size"] = 500
    param["epsilon_decay_step"] = 200
    param["gamma"] = 0.95
    param["train_freq"] = 1
    param["max_memory"] = 2000

    ep = 10

    env = ENV(AI_active=True , limit_speed=False, render=True)
    reward= train_Agent(ep, env, param, resume=True, is_train=False)

    plt.plot(range(len(reward)), reward)
    plt.xlabel("EP")
    plt.ylabel("Reward")
    plt.title("REWARD")
    plt.show()