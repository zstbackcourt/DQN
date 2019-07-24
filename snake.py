# -*- coding:utf-8 -*-
"""
DQN on snake env from LYW
@author: Weijie Shen
"""
import gym
import numpy as np
import tensorflow as tf
import dqn
from replay_buffer import ReplayBuffer
from env.SimpleEnv import SnakeEnv

def train(agent,buffer,env,num_episodes,max_steps,save_path):
    saver = tf.train.Saver(max_to_keep=5)
    checkpoint = tf.train.get_checkpoint_state(save_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find saved model")
    for i in range(num_episodes):
        cur_state = env.reset()[0]
        # print(cur_state,cur_state.shape)
        sum_reward = 0
        for step in range(max_steps):
            # print(step)
            # env.render()
            action = agent.get_action(cur_state)
            next_state, reward, done, info = env.step(action)
            next_state = next_state[0]
            done = done[0]
            # print(next_state, reward, done,type(next_state), type(reward), type(done))
            # sum_reward += reward
            if done:
                # print(done)
                sum_reward += reward
                buffer.add(state=cur_state, action=action, reward=reward, new_state=next_state, done=done)
                print("Episode {} finished after {} timesteps,reward sum {}".format(i, step + 1,sum_reward))
                break
            sum_reward += reward
            # print("action{}".format(action))
            buffer.add(state=cur_state, action=action, reward=reward, new_state=next_state, done=done)
            cur_state = next_state
        agent.epsilon_decay()
        print("当前的epsilon为{}".format(agent.epsilon))
        agent.learn(buffer=buffer,num_steps=128,batch_size=256)
        if i%20 == 0 and i>0:
            saver.save(sess, save_path)
            print("save model successfully!")

if __name__ == "__main__":
    env = SnakeEnv(gameSpeed=5,train_model=True)
    save_path = "./snake/model"
    # ob = env.reset()
    # print(ob,type(ob),ob.shape)
    buffer = ReplayBuffer(buffer_size=8192)
    sess = tf.Session()
    agent = dqn.DQNAgent(sess=sess,
                         epsilon = 0.9,
                         epsilon_anneal = 0.01,
                         end_epsilon = 0.1,
                         lr = 0.001,
                         gamma = 0.9,
                         state_size = 3,
                         action_size = 4,
                         name_scope = "dqn")
    sess.run(tf.global_variables_initializer())
    train(agent=agent,buffer=buffer,env=env,num_episodes=10000,max_steps=100,save_path=save_path)