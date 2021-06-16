import random
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt


from ENV import ENV

tf.disable_eager_execution()

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

        self.built_model()

    def built_model(self, scope="policy_network"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [None, 12], name='state')
            self.reward = tf.placeholder(tf.float32, [None], name='reward')
            self.action = tf.placeholder(tf.int32, [None], name='action')

            fc1 = tf.layers.dense(
                inputs = self.state,
                units = 128, 
                activation=tf.nn.relu,
                name = "FC1")
            fc2 = tf.layers.dense(
                inputs = fc1,
                units = 128, 
                activation=tf.nn.relu,
                name = "FC2")
            fc3 = tf.layers.dense(
                inputs = fc2,
                units = 128, 
                activation=tf.nn.relu,
                name = "FC3")

            self.prediction = tf.layers.dense(
                inputs = fc3,
                units = 4, 
                activation=None,
                name = "FC4")

            onehot_action = tf.one_hot(self.action, 4)
            self.action_prediction = tf.reduce_sum(tf.multiply(self.prediction, onehot_action), axis=1)

            self.losses = tf.squared_difference(self.reward, self.action_prediction)
            self.loss = tf.reduce_mean(self.losses)

            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

    def act(self, episode, state, sess):
        if self.is_train:
            if self.epsilon_decay_step == 0:
                self.epsilon = 0.01
            else:
                self.epsilon = self.epsilons[min(episode, self.epsilon_decay_step-1)]
        else:
            self.epsilon = 0

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        qValues = sess.run(self.prediction, {self.state: state})
        return np.argmax(qValues[0])

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) > self.max_memory:
                self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def train(self, step, sess):
        if self.batch_size > len(self.memory) or step % self.train_freq != 0:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        state_batch = np.array([i[0] for i in minibatch])
        action_batch = np.array([i[1] for i in minibatch])
        reward_batch = np.array([i[2] for i in minibatch])
        next_state_batch = np.array([i[3] for i in minibatch])
        done_batch = np.array([i[4] for i in minibatch])

        q_values_next = sess.run(self.prediction, {self.state: next_state_batch.reshape(self.batch_size, self.state_space)})

        targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * self.gamma * np.amax(q_values_next, axis=1)

        feed_dict = {self.state: state_batch.reshape(self.batch_size, self.state_space), self.reward: targets_batch, self.action: action_batch}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)

        return loss

def train_Agent(episode, env, param, resume=True, is_train=True):
    sum_reward = []
    step = 0
    running_reward = None
    agent = Agent(env, param, is_train=is_train)
    model_path = "model/DQN/model.ckpt"
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if resume:
        saver.restore(sess, model_path)

    for ep in range(episode):
        if not env.running:
            break
        state = env.reset()
        state = np.reshape(state, (1, env.state_space))
        score = 0
        while True:
            step += 1
            action = agent.act(ep, state, sess)
            if action == 0:
                env.go_up()
            if action == 1:
                env.go_down()
            if action == 2:
                env.go_left()
            if action == 3:
                env.go_right()
            next_state, reward, done = env.game_run()
            score += reward
            next_state = np.reshape(next_state, (1, env.state_space))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.train(step, sess)

            env.show()

            if done:
                running_reward = (score if running_reward is None else running_reward * 0.99 + score * 0.01)
                break
        print("end state: " + str(state))
        print("ep: " + str(ep+1) + " epsilon: " + str(agent.epsilon) + " reward: " + str(score) + " running reward: " + str(running_reward) + " loss: " + str(loss))
        sum_reward.append(running_reward)

        if ep % 20:
            saver.save(sess, model_path)
    sess.close()
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

    env = ENV(AI_active=True , limit_speed=True, render=True)
    reward = train_Agent(ep, env, param, resume=True, is_train=False)

    plt.plot(range(len(reward)), reward)
    plt.xlabel("EP")
    plt.ylabel("Reward")
    plt.title("REWARD")
    plt.show()
