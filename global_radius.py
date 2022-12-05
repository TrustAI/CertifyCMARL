import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from ma_gym.wrappers import Monitor
from torchsummary import summary
from collections import Counter
from itertools import zip_longest
from scipy.stats import norm, binom_test
from scipy.stats import qmc, norm, truncnorm
import tensorflow_probability as tfp
from scipy import stats
from copy import deepcopy
import gym
from gym.core import Wrapper
from pickle import dumps, loads
from collections import namedtuple
tfd = tfp.distributions
from statsmodels.stats.multitest import fdrcorrection
from  statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint,multinomial_proportions_confint
import sys, threading
from sklearn.preprocessing import normalize
sys.setrecursionlimit(10**7) # max depth of recursion
threading.stack_size(2**27)  # new thread will get stack of such size
class WithSnapshots(Wrapper):
    """
    Creates a wrapper that supports saving and loading environemnt states.
    Required for planning algorithms.

    This class will have access to the core environment as self.env, e.g.:
    - self.env.reset()           #reset original env
    - self.env.ale.cloneState()  #make snapshot for atari. load with .restoreState()
    - ...

    You can also use reset, step and render directly for convenience.
    - s, r, done, _ = self.step(action)   #step, same as self.env.step(action)
    - self.render(close=True)             #close window, same as self.env.render(close=True)
    """

    def get_snapshot(self, render=False):
        """
        :returns: environment state that can be loaded with load_snapshot 
        Snapshots guarantee same env behaviour each time they are loaded.

        Warning! Snapshots can be arbitrary things (strings, integers, json, tuples)
        Don't count on them being pickle strings when implementing MCTS.

        Developer Note: Make sure the object you return will not be affected by 
        anything that happens to the environment after it's saved.
        You shouldn't, for example, return self.env. 
        In case of doubt, use pickle.dumps or deepcopy.

        """
        if render:
            self.render()  # close popup windows since we can't pickle them
            self.close()
            
        if self.unwrapped.viewer is not None:
            self.unwrapped.viewer.close()
            self.unwrapped.viewer = None
        return dumps(self.env)

    def load_snapshot(self, snapshot, render=False):
        """
        Loads snapshot as current env state.
        Should not change snapshot inplace (in case of doubt, deepcopy).
        """

        assert not hasattr(self, "_monitor") or hasattr(
            self.env, "_monitor"), "can't backtrack while recording"

        if render:
            self.render()  # close popup windows since we can't load into them
            self.close()
        self.env = loads(snapshot)

    
def lower_confidence_bound( NA, N, alpha) :
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
def confidence_bound( N,alpha) :    
    return multinomial_proportions_confint( N, alpha=2 * alpha)
class QNet(nn.Module):
    def __init__(self, observation_space, action_space, recurrent=False):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        self.recurrent = recurrent
        self.hx_size = 32
        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i].shape[0]
            setattr(self, 'agent_feature_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 64),
                                                                            nn.ReLU(),
                                                                            nn.Linear(64, self.hx_size),
                                                                            nn.ReLU()))
            if recurrent:
                setattr(self, 'agent_gru_{}'.format(agent_i), nn.GRUCell(self.hx_size, self.hx_size))
            setattr(self, 'agent_q_{}'.format(agent_i), nn.Linear(self.hx_size, action_space[agent_i].n))
    def forward(self, obs, hidden):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        next_hidden = [torch.empty(obs.shape[0], 1, self.hx_size)] * self.num_agents
        for agent_i in range(self.num_agents):
            x = getattr(self, 'agent_feature_{}'.format(agent_i))(obs[:, agent_i, :])
            if self.recurrent:
                x = getattr(self, 'agent_gru_{}'.format(agent_i))(x, hidden[:, agent_i, :])
                next_hidden[agent_i] = x.unsqueeze(1)
            q_values[agent_i] = getattr(self, 'agent_q_{}'.format(agent_i))(x).unsqueeze(1)
        return torch.cat(q_values, dim=1), torch.cat(next_hidden, dim=1)
    def sample_action(self, obs, hidden, epsilon):
        out, hidden = self.forward(obs, hidden)
        mask = (torch.rand((out.shape[0],)) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1],))
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).float()
        action[~mask] = out[~mask].argmax(dim=2).float()
        return action, hidden
    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.hx_size))
class NoisyObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, sigma):
        super().__init__(env)
        self.sigma = sigma
    def observation(self, obs):
        return obs + np.array([np.zeros(np.array(obs).shape[1]),self.sigma*np.random.standard_normal(size=np.array(obs).shape[1])])
        #return obs + self.sigma*np.random.standard_normal(size=np.array(obs).shape)
        #return obs + np.array([self.sigma*np.random.standard_normal(size=np.array(obs).shape[1]),np.zeros(np.array(obs).shape[1])])
def possible_action(q,env,env2,state, hidden, sigma,alpha,sampling_num,n=4):
        hidden_box=[]
        action_box={}
        for iter in range(sampling_num):
            state_new = state+sigma*np.random.standard_normal(size=np.array(state).shape)
            #state_new =state+sample_noise[iter].reshape(np.array(state).shape)
            action_n, hidden_n = q.sample_action(torch.Tensor(state_new).unsqueeze(0), hidden, epsilon=0)
            for i in range(env.n_agents):
                action_box.setdefault(i,[]).append(action_n[0][i].numpy().tolist())
            hidden_box.append(hidden_n.numpy())
        action=[]
        for i in action_box:
            action.append(max(action_box[i],key = action_box[i].count))
        snapshot2 = env.get_snapshot()
        env2.load_snapshot(snapshot2)
        next_state, reward, done, info = env2.step(action)
        if_list=[]
        
        action_count={}
        for j in range(env.n_agents):
            h= np.array(action_box[j]).tolist()
            for i in range(5):
                action_count.setdefault(j,[]).append(h.count(i))
        for i in range(env.n_agents):
            Q_target=[]
            for j in range(5):
                env2.load_snapshot(snapshot2)
                action_new=action
                action_new[i]=j
                _,rew_j,_,_= env2.step(action_new)
                Q_target.append(sum(rew_j))
            otp=normalize(np.array(action_count[i]).reshape(1,-1),'max')
            baseline = torch.sum(torch.tensor(otp)* torch.tensor(Q_target)).detach()
            ifl=max(sum(reward) - baseline,0)
            if_list.append(ifl)
        hidden=[]
        radius_n=[]
        action_list={}
        hidden_list={}
        for i in action_count:
            pv1=stats.binom_test(max(action_count[i]), n=sum(sorted(action_count[i])[-2:]), p=0.5)
            index=action_box[i].index(max(action_box[i],key = action_box[i].count))
             
            if i >0:
                for b in action_list[i-1]:
                    m=0
                    action_list.setdefault(i,[]).append(b+[np.argmax(action_count[i])])
                    hidden_list.setdefault(i,[]).append(hidden_list[i-1][m]+[hidden_box[index][0,i,:]])
                    m=m+1
            else:
                action_list.setdefault(i,[]).append([np.argmax(action_count[i])])
                hidden_list.setdefault(i,[]).append([hidden_box[index][0,i,:]])
            if pv1*if_list[i] > alpha:
                a1 = sorted(set(action_box[i]),key = action_box[i].count)[-2]
                index1=action_box[i].index(a1)
                #radiusc= sigma * (norm.ppf(max(agent_pro[:,0])+sorted(agent_pro[:,0])[-2]))####top 2 probability
                radiusc= sigma * lower_confidence_bound(sum(sorted(action_count[i])[-2:]),sampling_num,alpha)
                if i >0:
                    for b in action_list[i-1]:
                        m=0
                        action_list.setdefault(i,[]).append(b+[a1])
                        hidden_list.setdefault(i,[]).append(hidden_list[i-1][m]+[hidden_box[index1][0,i,:]])
                        m=m+1
                else:
                    action_list.setdefault(i,[]).append([a1])
                    hidden_list.setdefault(i,[]).append([hidden_box[index1][0,i,:]])
            else:
                #radiusc = sigma/2 * (norm.ppf(max(agent_pro[:,0]))-norm.ppf(sorted(agent_pro[:,0])[-2]))
                radiusc = sigma * lower_confidence_bound(max(action_count[i]),sampling_num,alpha)
            radius_n.append(radiusc)
        #actionl=[(a,b,c,d)for a in action_list[0] for b in action_list[1] for c in action_list[2] for d in action_list[3]]
        action_l=action_list[list(action_list.keys())[-1]]
        
        hidden =np.asarray(hidden_list[list(hidden_list.keys())[-1]])
        
        radius = min(radius_n)
        return radius,action_l,hidden
def update_dict(dic, k, v):
        dic[k] = v if k not in dic else min(dic[k], v)
Rtot= 10000
def tree_expand(q,env,env2,done,radius,state,hidden,Rtot,R,sampling_num,alpha,sigma):
    if R>=Rtot:
        Rtot = min(R,Rtot)
        print(min(R,Rtot),radius)
        return 0
    if done==[True,True]:
        Rtot = min(R,Rtot)
        print('Rtot is',Rtot)
        print(min(R,Rtot),radius)
        return 0
    radius_new,action_list,hidden_list = possible_action(q,env,env2,state, hidden, sigma,alpha,sampling_num)
    if action_list==[]:
        Rtot = min(R,Rtot)
        print(min(R,Rtot),radius)
        return 0
    radius=min(radius,radius_new)
    print('len',len(action_list))
    #env2=deepcopy(env)
    snapshot = env.get_snapshot()
    i=0
    
    for a in action_list:
        #env2=deepcopy(env)
        env.load_snapshot(snapshot)
        next_state, reward, done, info = env.step(a)
        #reward = max(sum(reward), 0)
        reward = sum(reward)
        print(reward)
        tree_expand(q,env,env2,done,radius,next_state,torch.Tensor([hidden_list[i]]),Rtot,R+reward,sampling_num,alpha,sigma)
        i=i+1
sigma = 0
#env =  NoisyObsWrapper(gym.make("ma_gym:Switch2-v0"), sigma) 
env = WithSnapshots(gym.make('ma_gym:Switch4-v0'))
env2 = WithSnapshots(gym.make('ma_gym:Switch4-v0'))
#env = WithSnapshots(gym.make('ma_gym:Checkers-v0'))
#env2 = WithSnapshots(gym.make('ma_gym:Checkers-v0'))
#env =  NoisyObsWrapper(gym.make("ma_gym:Checkers-v0"), sigma)
#env = Monitor(env, directory='recordings/', force=True)
episodes = 10
done_n = [False for _ in range(env.n_agents)]
ep_reward = 0
obs = []
reward_tot =[]
state_n = []
labels_n =[]
rew=0
obs_n = env.reset()
#q = torch.load('minimal-marl/qmix_checkers')
#q = torch.load('minimal-marl/vdnma_gym:Switch4-v0')
#q = torch.load('minimal-marl/vdn')
q = torch.load('minimal-marl/qmix_Switch4-v0')
score = 0
action_num = env.action_space
sampling_num = 100
reward_n = 0
state = env.reset()
done = [False for _ in range(env.n_agents)]
state_n.append(state)
i=0
radius =1000

R=0
alpha=0.05
with torch.no_grad():
    hidden = q.init_hidden()
    tree_expand(q,env,env2,done,radius,state,hidden,Rtot,R,sampling_num,alpha,sigma)
#torch.save(reward_tot,'qmix_agent_defended'+ str(sampling_num)+str(sigma) + '.pth')
#torch.save(radius1,'qmix_agent1'+ str(sampling_num)+'_'+str(sigma) + '.pth')
#torch.save(radius2,'qmix_agent2'+ str(sampling_num)+'_'+str(sigma) + '.pth')
env.close()
#test_agent1,true_1= fdrcorrection(p1)
#test_agent2,true_2= fdrcorrection(p2)
#test_agent1= multipletests(p1,method='holm')
#test_agent2= multipletests(p2,method='holm')





