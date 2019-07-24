# -*- coding:utf-8 -*-
"""
DQN agent
@author: Weijie Shen
"""
import numpy as np
import tensorflow as tf
import tf_utils

class DQNAgent(object):
    def __init__(self,sess,epsilon,epsilon_anneal,end_epsilon,lr,gamma,state_size,action_size,name_scope):
        """

        :param sess:
        :param epsilon: e-greedy探索的系数
        :param epsilon_anneal: epsilon的线性衰减率
        :param end_epsilon: 最低的探索比例
        :param lr: learning rate
        :param gamma: 折扣率
        :param state_size: observation dim
        :param action_size: action dim
        :param name_scope: 命名域
        """
        self.sess = sess
        self.epsilon = epsilon
        self.epsilon_anneal = epsilon_anneal
        self.end_epsilon = end_epsilon
        self.lr = lr
        self.gamma = gamma
        self.state_size = state_size
        self.action_size = action_size
        self.name_scope = name_scope
        self.qnetwork()

    def qnetwork(self):
        """
        创建Q network
        :return:
        """
        with tf.variable_scope(self.name_scope):
            self.state_input = tf.placeholder(tf.float32, [None, self.state_size]) # 状态输入
            self.action = tf.placeholder(tf.int32, [None]) # 动作输入
            self.target_q = tf.placeholder(tf.float32, [None]) # target Q

            fc1 = tf_utils.fc(self.state_input, n_output=16, activation_fn=tf.nn.relu)
            fc2 = tf_utils.fc(fc1, n_output=32, activation_fn=tf.nn.relu)
            fc3 = tf_utils.fc(fc2, n_output=16, activation_fn=tf.nn.relu)
            self.q_values = tf_utils.fc(fc3, self.action_size, activation_fn=None)
            # 动作用one-hot编码
            action_mask = tf.one_hot(self.action, self.action_size, 1.0, 0.0)
            # 预测的q
            q_value_pred = tf.reduce_sum(self.q_values * action_mask, 1)
            # q network的loss
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.target_q, q_value_pred)))
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = self.optimizer.minimize(self.loss)

    def get_action_values(self, state):
        actions = self.sess.run(self.q_values, feed_dict={self.state_input: [state]})
        return actions

    def get_optimal_action(self,state):
        """
        最优action就是对应的最大value的action
        :param state:
        :return:
        """
        actions = self.sess.run(self.q_values, feed_dict={self.state_input: [state]})
        return actions.argmax()

    def get_action(self,state):
        """
        用e-greedy策略选择与环境交互的action
        :param state:
        :return:
        """
        if np.random.random() < self.epsilon:
            # 以epsilon的概率随机选择一个动作
            return np.random.randint(0,self.action_size)
        else:
            return self.get_optimal_action(state)

    def epsilon_decay(self):
        """
        epsilon衰减
        :return:
        """
        if self.epsilon > self.end_epsilon:
            self.epsilon -= self.epsilon_anneal

    def learn(self,buffer,num_steps,batch_size):
        if buffer.size()<batch_size:
            return
        for step in range(num_steps):
            minibatch = buffer.get_batch(batch_size=batch_size)
            state_batch = [data[0] for data in minibatch]
            action_batch = [data[1] for data in minibatch]
            reward_batch = [data[2] for data in minibatch]
            next_state_batch = [data[3] for data in minibatch]
            done_batch = [data[4] for data in minibatch]

            q_values = self.sess.run(self.q_values, feed_dict={self.state_input: next_state_batch})
            max_q_values = q_values.max(axis=1)

            # 计算target q value
            target_q = np.array(
                [data[2] + self.gamma*max_q_values[i]*(1-data[4]) for i,data in enumerate(minibatch)]
            )
            target_q = target_q.reshape([batch_size])

            # 最小化TD-error,即训练
            l , _ = self.sess.run([self.loss,self.train_op],feed_dict={
                self.state_input:state_batch,
                self.target_q:target_q,
                self.action:action_batch
            })