import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

from ENV import ENV

tf.disable_eager_execution()

class Agent:
    def __init__(self, env, param):
        self.state_space = env.state_space
        self.action_space = env.action_space

        self.learning_rate = param["learning_rate"]
        self.gamma = param["gamma"]

        self.built_model()

    def built_model(self, scope="policy_network"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [None, 12], name='state')
            self.reward = tf.placeholder(tf.float32, [None], name='reward')
            self.action = tf.placeholder(tf.int32, [None], name='action')

            fc1 = tf.layers.dense(
                inputs = self.state,
                units = 128, 
                activation=tf.math.tanh,
                name = "FC1")
            fc2 = tf.layers.dense(
                inputs = fc1,
                units = 128, 
                activation=tf.math.tanh,
                name = "FC2")
            fc3 = tf.layers.dense(
                inputs = fc2,
                units = 128, 
                activation=tf.math.tanh,
                name = "FC3")

            logits = tf.layers.dense(
                inputs = fc3,
                units = 4, 
                activation=None,
                name = "FC4")

            self.action_prob = tf.nn.softmax(logits)

            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.action, logits=logits)
            self.loss = tf.reduce_mean(tf.multiply(neg_log_prob, self.reward))

            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

    def act(self, state, sess):
        action_prob = sess.run(self.action_prob, feed_dict={self.state: state})
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

    def train(self, state_list, reward_list, action_list, sess):
        state_batch = np.vstack(state_list)
        action_batch = np.array(action_list)
        reward_batch = np.array(reward_list)

        discounted_epr = self.discount_reward(reward_batch)

        feed_dict = {self.state: state_batch, self.reward: discounted_epr, self.action: action_batch}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)

        return loss

def train_Agent(episode, env, param, resume=True):
    sum_reward = []
    running_reward = None
    agent = Agent(env, param)
    model_path = "model/PolicyGradients/model.ckpt"
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if resume:
        saver.restore(sess, model_path)

    for ep in range(episode):
        if not env.running:
            break
        state_list, reward_list, action_list = [], [], []
        state = env.reset()
        state = np.reshape(state, (1, env.state_space))
        score = 0
        while True:
            action = agent.act(state, sess)

            action_list.append(action)
            state_list.append(state)

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
            state = next_state

            reward_list.append(reward)

            env.show()

            if done:
                agent.train(state_list, reward_list, action_list, sess)
                running_reward = (score if running_reward is None else running_reward * 0.99 + score * 0.01)
                break
        print("end state: " + str(state))
        print("ep: " + str(ep+1) + " reward: " + str(score) + " runing reward: " + str(running_reward))
        sum_reward.append(running_reward)

        if ep % 20:
            saver.save(sess, model_path)
    sess.close()
    return sum_reward

if __name__ == "__main__":
    param = dict()
    param["learning_rate"] = 0.00025
    param["gamma"] = 0.95

    ep = 10

    env = ENV(AI_active=True , limit_speed=True, render=True)
    reward = train_Agent(ep, env, param, resume=True)

    plt.plot(range(len(reward)), reward)
    plt.xlabel("EP")
    plt.ylabel("Reward")
    plt.title("REWARD")
    plt.show()
