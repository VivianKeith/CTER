"""
This is a Curiosity-Tuned-PER(CTPER) implementation.
Curiosity here is a simple version of [Large-Scale Study of Curiosity-Driven Learning](https://arxiv.org/abs/1808.04355)
"""

import time
import random
import math
import os.path as osp
import os
from turtle import color

from matplotlib.pyplot import savefig

import numpy as np
import tensorflow as tf

from tqdm import tqdm

from rltookit.config import DEFAULT_SEED
from rltookit.log import EpochLogger, colorize


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then upward propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class CuriosityTunedPerDqn:
    def __init__(
            self,
            action_dim,
            obs_dim,
            q_net_hidden_sizes=(128,),
            lr=0.01,
            gamma=0.98,
            soft_update_tau=0.001,
            initial_epsilon=0.0,
            final_epsilon=0.95,
            epsilon_increment=20000,
            replace_target_iter=300,
            memory_size=10000,
            batch_size=128,
            per=True,
            curiosity_tune_reward=False,
            curiosity_tune_per=False,
            curiosity_eta=0.1,
            curiosity_update_freq=1000,
            dynamic_net_hidden_sizes=(128,),
            sess=None,
            tflog_path='tflog',
            save_freq=100
    ):
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.q_net_hidden_sizes = q_net_hidden_sizes
        self.lr = lr
        self.gamma = gamma
        self.soft_update_tau=soft_update_tau  # a ratio in (0,1], where 1 means hard replacement, the bigger, the harder
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_increment = epsilon_increment
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.per = per  # decide to use per or not
        self.curiosity_tune_reward = curiosity_tune_reward
        self.curiosity_tune_per = curiosity_tune_per
        self.curiosity_eta = curiosity_eta
        self.curiosity_update_freq = curiosity_update_freq
        self.dynamic_net_hidden_sizes=dynamic_net_hidden_sizes

        # validate params
        if not per and curiosity_tune_per:
            raise ValueError(
                colorize('[per] is set to False, but [curiosity_tune_per] is set to True', color='red')
            )
            
        if curiosity_tune_per and curiosity_tune_reward:
            raise ValueError(
                colorize("[curiosity_tune_reward] and [curiosity_tune_per] can not set to True together.", color='red')
            )
        
        if soft_update_tau == 0:
            raise ValueError(
                colorize("soft update ratio can not be set to 0.", color='red')
            )
            
        # build memory buffer
        if self.per:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, self.obs_dim*2+2))

        # setup session
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        # total learning step
        self.learn_step_counter = 0
        self.memory_counter = 0

        # step 1: Setup tensorboard summary
        self.summary_writer = tf.summary.FileWriter(osp.join(tflog_path, 'summary'), self.sess.graph)
        self.summary_ops = []

        # Step 2: build networks
        self._build_nets()      # When build graph, adding some summary ops to self.summary_ops

        # Step 3: setup model saving
        # must create saver after build graph, otherwise ValueError: No variables to save
        self.model_saver = tf.train.Saver(max_to_keep=2)
        self.ckpt_path = osp.join(tflog_path, 'ckpt')
        os.makedirs(self.ckpt_path, exist_ok=True)
        self.save_freq = save_freq

    def _soft_update_q_target(self, target_params, eval_params):
            new_target_params = self.soft_update_tau * eval_params + (1-self.soft_update_tau) * target_params
            return new_target_params

    def _mlp(self, x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None, trainable=True):
        for h in hidden_sizes[:-1]:
            x = tf.layers.dense(x, units=h, activation=activation, trainable=trainable)
        return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, trainable=trainable)
    
    def _build_nets(self):

        self.tfs = tf.placeholder(tf.float32, [None, self.obs_dim], name="s")    # input State
        self.tfa = tf.placeholder(tf.int32, [None, ], name="a")              # input Action
        self.tfr = tf.placeholder(tf.float32, [None, ], name="ext_r")        # extrinsic reward
        self.tfs_ = tf.placeholder(tf.float32, [None, self.obs_dim], name="s_")  # input Next State

        if self.per:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        
        # build dqn module
        if self.curiosity_tune_reward or self.curiosity_tune_per:
            self.tf_curiosity = self._build_dynamics_curiosity(self.tfs, self.tfa, self.tfs_)
            self._build_dqn(self.tfs, self.tfa, self.tfr, self.tfs_, curiosity=self.tf_curiosity)
        else:
            self._build_dqn(self.tfs, self.tfa, self.tfr, self.tfs_, curiosity=None)
        
        # build tf summary op
        if len(self.summary_ops) > 0:
            self.merged_summary_op = tf.summary.merge(self.summary_ops)
        else:
            raise ValueError('No summary_op is set!. Add some and try again!')

    def _build_dynamics_curiosity(self, s, a, s_):
        with tf.variable_scope("dynamic_curiosity_net"):
            float_a = tf.expand_dims(tf.cast(a, dtype=tf.float32, name="float_a"), axis=1, name="2d_a")
            sa = tf.concat((s, float_a), axis=1, name="sa")
            encoded_s_ = s_                # here we use s_ as the encoded s_
            dyn_s_ = self._mlp(sa, hidden_sizes=list(self.dynamic_net_hidden_sizes)+[self.obs_dim], activation=tf.nn.relu)   # predicted s_

        with tf.name_scope("curiosity"):
            curiosity = tf.reduce_sum(tf.square(encoded_s_ - dyn_s_), axis=1)
            self.curiosity_summary_op = tf.summary.histogram('curiosity', curiosity)
        
        # It is better to reduce the learning rate in order to stay curious
        self.curiosity_train_op = tf.train.RMSPropOptimizer(self.lr, name="dynamic_curiosity_train_opt").minimize(
            tf.reduce_mean(curiosity))
        
        return curiosity

    def _build_dqn(self, s, a, r, s_, curiosity=None):
        
        # curiosity tune reward
        if curiosity is not None and self.curiosity_tune_reward:
            r = tf.add(curiosity, r, name="total_reward")

        with tf.variable_scope('eval_net'):
            self.q_eval = self._mlp(s, hidden_sizes=list(self.q_net_hidden_sizes)+[self.action_dim], activation=tf.nn.relu)   # shape=(batch_size, act_dim)
            self.summary_ops.append(tf.summary.histogram('q_eval', self.q_eval))

        with tf.variable_scope('target_net'):
            q_ = self._mlp(s_, hidden_sizes=list(self.q_net_hidden_sizes)+[self.action_dim], activation=tf.nn.relu, trainable=False)   # shape=(batch_size, act_dim)
            self.q_target = r + self.gamma * tf.reduce_max(q_, axis=1, name="Qmax_s_")  # shape=(batch_size, )
            self.summary_ops.append(tf.summary.histogram('q_target', self.q_target))

        with tf.variable_scope('q_wrt_a'):
            a_indices = tf.stack([tf.range(tf.shape(a)[0], dtype=tf.int32), a], axis=1) # shape=(batch_size, 2)
            q_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)   # shape=(batch_size, )
        
        t_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('target_replacement'):
            self.target_replace_op = [
                tf.assign(t, e) for t, e in zip(t_params, e_params)]

        with tf.variable_scope('loss'):
            if self.per:
                    self.abs_errors = tf.abs(self.q_target - q_wrt_a)   # batch TD errors, for updating transition's priority in Sumtree
                    self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, q_wrt_a))

                    if curiosity is not None and self.curiosity_tune_per:
                        self.abs_errors = tf.multiply((curiosity**self.curiosity_eta), self.abs_errors,
                                                name='curiosity_tune_per_abs_errors')
                    
                    self.summary_ops.append(tf.summary.histogram('abs_errors', self.abs_errors))
                    self.summary_ops.append(tf.summary.scalar('loss', self.loss))
            else:
                self.loss = tf.losses.mean_squared_error(labels=self.q_target, predictions=q_wrt_a)   # TD error
                self.summary_ops.append(tf.summary.scalar('loss', self.loss))

        self.dqn_train_op = tf.train.RMSPropOptimizer(self.lr, name="dqn_train_opt").minimize(self.loss)
    
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        if self.per:
            self.memory.store(transition)    # have high priority for newly arrived transition
        else:
            # replace the old memory with new memory
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def choose_action(self, observation, how_greedy='epsilon'):
        # to have batch dimension when feed into tf placeholder
        s = observation[np.newaxis, :]
        
        if how_greedy == 'epsilon':
            if np.random.uniform() < self.epsilon:
                # forward feed the observation and get q value for every actions
                actions_value = self.sess.run(self.q_eval, feed_dict={self.tfs: s})
                return np.argmax(actions_value)
            else:
                return np.random.randint(0, self.action_dim)
        
        if how_greedy == 'argmax':
            actions_value = self.sess.run(self.q_eval, feed_dict={self.tfs: s})
            return np.argmax(actions_value)
        else:
            raise ValueError('Specify parameter[how_greedy] to epsilon or argmax.')
    
    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)

        # sample batch memory from all memory
        if self.per:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            top = self.memory_size if self.memory_counter > self.memory_size else self.memory_counter
            sample_index = np.random.choice(top, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        bs = batch_memory[:, :self.obs_dim]
        ba = batch_memory[:, self.obs_dim]
        br = batch_memory[:, self.obs_dim + 1]
        bs_ = batch_memory[:, -self.obs_dim:]

        # run sess to update network params
        if self.per:
            _, abs_errors, self.loss_record, summary = self.sess.run(
                [self.dqn_train_op, self.abs_errors, self.loss, self.merged_summary_op],
                feed_dict={self.tfs: bs, self.tfa: ba, self.tfr: br, self.tfs_: bs_, self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)
        else:
            _, self.loss_record, summary = self.sess.run(
                [self.dqn_train_op, self.loss, self.merged_summary_op],
                feed_dict={self.tfs: bs, self.tfa: ba, self.tfr: br, self.tfs_: bs_})

        # delay training in order to stay curious
        if self.curiosity_tune_per or self.curiosity_tune_reward:
            if self.learn_step_counter % self.curiosity_update_freq == 0:
                _, summary = self.sess.run(
                    [self.curiosity_train_op, self.curiosity_summary_op],
                    feed_dict={self.tfs: bs, self.tfa: ba, self.tfs_: bs_}
                )
        
        self.learn_step_counter += 1
        self.summary_writer.add_summary(summary, self.learn_step_counter)
        self.epsilon = self.epsilon + (self.final_epsilon-self.epsilon) / self.epsilon_increment

        # saving ckpt model
        if self.learn_step_counter % self.save_freq == 0:
            self.save_ckpt()
    
    def save_ckpt(self):
        self.model_saver.save(
            sess=self.sess,
            save_path=osp.join(self.ckpt_path, 'dqnmodel.ckpt'),
            global_step=self.learn_step_counter
        )

    def restore_ckpt(self):
        self.model_saver.restore(
            sess=self.sess,
            save_path=tf.train.latest_checkpoint(self.ckpt_path)
        )

    def get_dqn_loss(self):
        if hasattr(self, 'loss_record'):
            return self.loss_record
        else:
            raise ValueError(
                colorize('Network has not been trained, NO loss recorded!')
            )


# training entry
def dqn_ctper(env_name='gym', difficulty=0, render=False, early_finish=False, target_return=100,
            episodes=100, max_episode_len=30000, update_after=10000, update_every=10, save_every=1, num_test_epochs=10,
            q_net_hidden_sizes=(64,), lr=0.005, gamma=0.9, replace_target_freq=500, memory_size=10000,
            batch_size=32, initial_epsilon=0.2, final_epsilon=0.95 , epsilon_increment=20000,
            per=False, curiosity_tune_per=False, curiosity_tune_reward=False, terminal_reward_reshape_ratio=1.0,
            curiosity_eta=0.1, curiosity_update_freq=1000, dynamic_net_hidden_sizes=(64,),
            **exp_configs):
    
    """DQN with Curiosity-Tuned-PER.

    Args:
        env_name (str, optional): env name, 'gym' | 'soccer' | 'combat'. Defaults to 'gym'.
        
        render (bool, optional): Whether to render env steps. Defaults to False.
        
        early_finish (bool, optional): Whether to early finish training based on
            if achieve target return. Defaults to False.
        
        target_return (int, optional): Value of target return, only for determining
            whether to early finish. Defaults to 100.
        
        episodes (int, optional): Number of episodes to run and train agent. Defaults to 100.
        
        max_episode_len (int, optional): Maximum length of trajectory / episode / rollout. Defaults to 2000.
        
        update_after (int, optional): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates. Defaults to 10000.
        
        update_every (int, optional): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1. Defaults to 10.
        
        save_every (int, optional): How often (in terms of gap between episodes)
            to save the current model(only for log util). Defaults to 1.
        
        num_test_epochs (int, optional): Number of episodes to test the policy. Defaults to 10.
        
        hidden_sizes (tuple, optional): Sizes of hidden layers in networks. Defaults to (64,).
        
        lr (float, optional): learning rate. Defaults to 0.005.
        
        gamma (float, optional): Reward discount factor. (Always between 0 and 1.). Defaults to 0.9.
        
        replace_target_freq (int, optional): How often (in terms of gap between gradient descents of current Q net)
            to replace target Q net by current Q net. Defaults to 500.
        
        memory_size (int, optional): Maximum length of memory replay buffer. Defaults to 10000.
        
        batch_size (int, optional): Minibatch size for SGD. Defaults to 32.

        initial_epsilon (float, optional): Initial value of epsilon for e-greedy. Defaults to 0.2.

        final_epsilon (float, optional): Final value of epsilon for e-greedy. Defaults to 0.95.
        
        epsilon_increment (float, optional): 1/epsilon_increment is the increment for increasing epsilon
            to the final value. Defaults to 20000.
        
        per (bool, optional): Whether to use Prioritized Experience Replay. Defaults to True.
        
        curiosity_tune_per (bool, optional): Whether to use dynamic curiosity to tune PER. Defaults to False.

        curiosity_tune_reward (bool, optional): Whether to use dynamic curiosity to tune reward. Defaults to False.

        terminal_reward_reshape_ratio (float, optional): A ratio of scaled terminal reward to original terminal reward. Default to 1.0.

        curiosity_eta (float, optional): A hyperparameter, the exponent of curiosity, to determine how serious 
            is the curiosity-tuning. Defaults to 0.1. (Always between 0 and 1.)

        curiosity_update_freq (int, optional): How often (in terms of gap between gradient descents of current Q net)
            to update dynamic curiosity network. Defaults to 1000. 

    """
    
    # init a logger
    logger = EpochLogger(**exp_configs)
    logger.save_config(locals())

    # set random seed
    seed = exp_configs.get('seed', DEFAULT_SEED)
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # build env
    if env_name == 'gym':
        import gym
        env = gym.make('MountainCar-v0').unwrapped  # unwrap the time limit
        env.seed(seed)
        action_dim = 3
        obs_dim = 2
    
    elif env_name == 'soccer':
        from soccer_env.env_Soccer import EnvSoccer
        env = EnvSoccer(difficulty=difficulty, max_steps_in_episode=max_episode_len)
        action_dim = 4**3
        obs_dim = 40

    elif env_name == 'combat':
        pass
    
    # clear variables in graph, otherwise ValueError: Variable xxx already exists, disallowed.
    tf.reset_default_graph()
    
    # create sesstion
    sess = tf.Session()

    dqn = CuriosityTunedPerDqn(
        action_dim=action_dim,
        obs_dim=obs_dim,
        q_net_hidden_sizes=q_net_hidden_sizes,
        lr=lr,
        gamma=gamma,
        initial_epsilon=initial_epsilon,
        final_epsilon=final_epsilon,
        epsilon_increment=epsilon_increment,
        replace_target_iter=replace_target_freq,
        memory_size=memory_size,
        batch_size=batch_size,
        per=per,
        curiosity_tune_reward=curiosity_tune_reward,
        curiosity_tune_per=curiosity_tune_per,
        curiosity_eta=curiosity_eta,
        curiosity_update_freq=curiosity_update_freq,
        dynamic_net_hidden_sizes=dynamic_net_hidden_sizes,
        sess=sess,
        tflog_path=osp.join(logger.output_dir, 'tflog')
    )

    sess.run(tf.global_variables_initializer())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'s': dqn.tfs}, outputs={'q_eval': dqn.q_eval})
    
    # Prepare for interaction with environment
    total_steps = 0
    start_time = time.time()
    
    # Main loop: training agent during epochs
    for episode in range(episodes):
        observation, episode_return, episode_len = env.reset(), 0, 0
        # training
        # for step in tqdm(range(steps_per_epoch), desc="epoch %s training"%epoch):
        while True:
            if render:
                env.render()

            # Get the action using agent's policy
            # action = dqn.choose_action(observation, how_greedy='epsilon')    # e-greedy action for train
            action = dqn.choose_action(observation)
            
            # Step the env
            observation_, reward, done, _ = env.step(action)
            
            # reshape MountainCar rewards
            if env_name == 'gym':
                if reward == -1:
                    reward = 0
                if done: reward = 1
            
            if env_name == 'soccer':
                if done: reward *= terminal_reward_reshape_ratio
            
            episode_return += reward
            episode_len += 1
            total_steps += 1
            
            print('episode:', episode, 'episode len:', episode_len, 'total steps:', total_steps)
        

            # Store experience to replay buffer
            dqn.store_transition(observation, action, reward, observation_)
            
            # Super critical! easy to overlook step: make sure to update to the most recent observation!
            observation = observation_
            
            # Update policy handling
            if total_steps >= update_after and total_steps % update_every == 0:
                for _ in range(update_every):
                    dqn.learn()
                    print('LOSS:', dqn.get_dqn_loss())
            
            # End of trajectory handling
            if done:
                print('episode %s finished:'%episode, 'episode len:', episode_len, 'total steps:', total_steps)
                logger.log_tabular('Episode', episode)
                logger.log_tabular('EpRet', episode_return)
                logger.log_tabular('EpLen', episode_len)
                logger.log_tabular('TotalInteracts', total_steps)
                logger.log_tabular('Time', time.time()-start_time)
                logger.dump_tabular()
                break
        
        if (episode+1) % save_every == 0:
            logger.save_state({}, None)

        # Early finish training
        if early_finish:
            if episode_return >= target_return:
                print(colorize("Training finished! After %s episodes, the episode return achieve %s!"%(episode, target_return), color="red", bold=True))
                break


if __name__ == "__main__":
    dqn_ctper()
