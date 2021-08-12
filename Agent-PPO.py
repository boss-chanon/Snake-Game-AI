import numpy as np
import matplotlib.pyplot as plt
import copy

import tensorflow as tf
from tensorflow.keras.activations import tanh, relu, softmax, gelu
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import backend as K

from ENV import ENV

tf.compat.v1.disable_eager_execution()

class Critic:
    def __init__(self, state_space, action_space, learning_rate):
        X_input = Input(state_space)
        old_values = Input(shape=(1,))

        X = Dense(128, activation=gelu)(X_input)
        X = Dense(128, activation=gelu)(X)
        X = Dense(128, activation=gelu)(X)
        output = Dense(1, activation=None)(X)

        self.model = Model(inputs=[X_input, old_values], outputs=output)
        self.model.compile(loss=self.critic_PPO2_loss(old_values), optimizer=Adam(learning_rate=learning_rate))

    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2
            clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2
            
            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            #value_loss = K.mean((y_true - y_pred) ** 2)
            return value_loss
        return loss

    def predict(self, state):
        return self.model.predict([state, np.zeros((state.shape[0], 1))])

    def update(self, state, reward, values):
        c_loss = self.model.fit([state, values], reward, verbose=0, epochs=10)

        return np.sum(c_loss.history["loss"])

class Actor:
    def __init__(self, state_space, action_space, learning_rate):
        X_input = Input(state_space)
        self.action_space = action_space

        X = Dense(128, activation=tanh)(X_input)
        X = Dense(128, activation=tanh)(X)
        X = Dense(128, activation=tanh)(X)
        output = Dense(action_space, activation=softmax)(X)

        self.model = Model(inputs=X_input, outputs=output)
        self.model.compile(loss=self.PPO_loss, optimizer=Adam(learning_rate=learning_rate))

    def PPO_loss(self, y_true, y_pred):
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]

        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001

        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio =  K.exp(K.log(prob) - K.log(old_prob))

        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1-LOSS_CLIPPING, max_value=1+LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        
        total_loss = actor_loss - entropy

        return total_loss

    def predict(self, state):
        return self.model.predict(state)

    def update(self, state, advantages, predictions, actions):
        y_true = np.hstack([advantages, predictions, actions])

        loss = self.model.fit(state, y_true, verbose=0, epochs=10)
        return np.sum(loss.history["loss"])

class Agent:
    def __init__(self, env, param):
        self.state_space = env.state_space
        self.action_space = env.action_space

        self.learning_rate = param["learning_rate"]

        self.memory = []

        self.built_model()

    def save(self, path):
        self.actor.model.save(path+"actor.h5")
        self.critic.model.save(path+"critic.h5")

    def load(self, path):
        self.actor.model.load_weights(path+"actor.h5")
        self.critic.model.load_weights(path+"critic.h5")

    def built_model(self):
        self.actor = Actor(self.state_space, self.action_space, learning_rate=self.learning_rate)
        self.critic = Critic(self.state_space, self.action_space, learning_rate=self.learning_rate)

    def act(self, state):
        action_prob = self.actor.predict(state)[0]

        return np.random.choice(a=4, p=action_prob), action_prob

    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def train(self, states, actions, rewards, predictions, dones, next_states):
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        values = self.critic.predict(states)
        next_values = self.critic.predict(next_states)

        values = np.reshape(values, (len(values)))
        next_values = np.reshape(next_values, (len(next_values)))

        advantages, target = self.get_gaes(rewards, dones, values, next_values)

        c_loss = self.critic.update(states, target, values)
        a_loss = self.actor.update(states, advantages, predictions, actions)

        return a_loss, c_loss

def train_Agent(episode, env, param, resume=True):
    sum_reward = []
    scores = []
    step = 0
    running_reward = None
    running_reward_max = 100
    agent = Agent(env, param)
    model_path = "model/PPO/"

    if resume:
        agent.load(model_path)
    for ep in range(episode):
        if not env.running:
            break
        state_list, reward_list, action_list, prediction_list, done_list, next_state_list = [], [], [], [], [], []
        state = env.reset()
        state = np.reshape(state, (1, env.state_space))
        score = 0
        while True:
            action, prediction = agent.act(state)

            action_onehot = np.zeros([env.action_space])
            action_onehot[action] = 1

            action_list.append(action_onehot)
            state_list.append(state)
            prediction_list.append(prediction)

            next_state, reward, done = env.game_run(action)
            score += reward
            next_state = np.reshape(next_state, (1, env.state_space))
            state = next_state

            reward_list.append(reward)
            done_list.append(done)
            next_state_list.append(next_state)

            env.show()

            if done:
                scores.append(score)
                a_loss, c_loss = agent.train(state_list, action_list, reward_list, prediction_list, done_list, next_state_list)
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

    ep = 10

    env = ENV(AI_active=True , limit_speed=False, render=True)
    reward= train_Agent(ep, env, param, resume=True)

    plt.plot(range(len(reward)), reward)
    plt.xlabel("EP")
    plt.ylabel("Reward")
    plt.title("REWARD")
    plt.show()