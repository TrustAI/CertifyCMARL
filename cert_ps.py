import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ma_gym.wrappers import Monitor
from torchsummary import summary
from collections import Counter
from itertools import zip_longest
from scipy.stats import norm, binom_test
from scipy.stats import qmc, norm, truncnorm
from gym.core import Wrapper
import attacks
from pickle import dumps, loads
import tensorflow_probability as tfp
from tqdm import tqdm
from scipy import stats
tfd = tfp.distributions
from statsmodels.stats.multitest import fdrcorrection
from  statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint,multinomial_proportions_confint
from sklearn.preprocessing import normalize
import random
from models import QNet, model_setup
#from test_tree_our import possible_action,tree_expand
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(10)
random.seed(2022)
def sample_action(state, hidden):
    global q
    global env2
    global sigma
    global alpha
    global sampling_num 
    hidden_box=[]
    action_box={}
    noise=[ torch.FloatTensor(*(state.shape)).normal_(0, sigma).cuda()
                        for _ in tqdm(range(sampling_num)) ]
    for iter in range(sampling_num):
            #state_new = state+noise[iter]
            state_new =state+noise[iter]
            #action_n, hidden_n = q.sample_action(torch.Tensor(state_new).cuda().unsqueeze(0), hidden, epsilon=0)
            action_n, hidden_n = q.sample_action(state_new.unsqueeze(0), hidden, epsilon=0)
            for i in range(env.n_agents):
                action_box.setdefault(i,[]).append(action_n[0][i].cpu().numpy().tolist())
            hidden_box.append(hidden_n.cpu().numpy())
    action=[]
    for i in action_box:
        action.append(max(action_box[i],key = action_box[i].count))
    
    return action,action_box,hidden_box
        
def possible_action(env,state,hidden,state_shape,n=4):
    global q
    global env2
    global sigma
    global alpha
    global sampling_num 
    action,action_box,hidden_box= sample_action(state, hidden)
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
    for i in action_count: #choose agent i
        pv1=stats.binom_test(max(action_count[i]), n=sampling_num, p=0.5)
        index=action_box[i].index(max(action_box[i],key = action_box[i].count))
        #pv.setdefault(i,[]).append(pv1)  
        if i >0: ##in agent great than 1 append action 
            for b in action_list[i-1]:
                m=0
                action_list.setdefault(i,[]).append(b+[np.argmax(action_count[i])]) #previous action add new action
                hidden_list.setdefault(i,[]).append(hidden_list[i-1][m]+[hidden_box[index][0,i,:]])
                m=m+1
        else:
            action_list.setdefault(i,[]).append([np.argmax(action_count[i])])
            hidden_list.setdefault(i,[]).append([hidden_box[index][0,i,:]])
        if pv1 > alpha: #*if_list[i]
            a1=list(set(sorted(action_box[i],key = action_box[i].count)))[-2]
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

#Rtot= 10000
def tree_expand(env,done,radius,state,hidden,Rtot,R,action_tot):
    global q
    global env2
    global sigma
    global alpha
    global sampling_num
    if R>=Rtot:
        Rtot = min(R,Rtot)
        print(min(R,Rtot),radius)
        return 0
    print('Rtot is',Rtot)
    if done==[True,True]:
        Rtot = min(R,Rtot)
        print('Rtot is',Rtot)
        print(min(R,Rtot),radius)
        return 0
    if done==[True,True,True,True]:
        Rtot = min(R,Rtot)
        print('Rtot is',Rtot)
        print(min(R,Rtot),radius)
        return 0
    print(np.array(state).shape)
    state_shape= np.array(state).shape
    radius_new,action_list,hidden_list = possible_action(env,torch.Tensor(state).to(device), hidden,state_shape)
    if action_list==[]:
        Rtot = min(R,Rtot)
        print(min(R,Rtot),radius)
        return 0
    radius=min(radius,radius_new)
    print('len',len(action_list))
    
    snapshot = env.get_snapshot()
    
    for a in action_list:
        action_tot.append(a)
        print('action is',a)
        env.load_snapshot(snapshot)
        next_state, reward, done, info = env.step(a)
        #reward = max(sum(reward), 0)
        reward = sum(reward)
        index=action_list.index(a)
        print('this action',a)
        print('Ris',R)
        print('reward is',reward)
        torch.save(action_tot,'qmix'+ str(sigma) + '.pth')
        tree_expand(env,done,radius,next_state,torch.Tensor(hidden_list[index]).unsqueeze(0).to(device),Rtot,R+reward,action_tot)
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

class NoisyObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, sigma):
        super().__init__(env)
        self.sigma = sigma
    def observation(self, obs):
        return obs + np.array([np.zeros(np.array(obs).shape[1]),self.sigma*np.random.standard_normal(size=np.array(obs).shape[1])])
        #return obs + self.sigma*np.random.standard_normal(size=np.array(obs).shape)
        #return obs + np.array([self.sigma*np.random.standard_normal(size=np.array(obs).shape[1]),np.zeros(np.array(obs).shape[1])])
#sigma = 0.03
#env_name='Switch4-v0'
#env_name='TrafficJunction10-v0'
env_name='Checkers-v0'
#env =  NoisyObsWrapper(gym.make("ma_gym:Switch2-v0"), sigma) 
env=WithSnapshots(gym.make('ma_gym:'+env_name))
env2=WithSnapshots(gym.make('ma_gym:'+env_name))
env.seed(2022)
env2.seed(2022)
#env = gym.make('ma_gym:Switch4-v0')
#env =  NoisyObsWrapper(gym.make("ma_gym:Checkers-v0"), sigma)
#env = Monitor(env, directory='recordings/', force=True)
episodes = 1
done_n = [False for _ in range(env.n_agents)]
ep_reward = 0
obs = []
reward_tot =[]
state_n = []
labels_n =[]
rew=0
obs_n = env.reset()

#q = torch.load('minimal-marl/qmix_checkers')
q = torch.load('minimal-marl/vdn')
#q = model_setup('minimal-marl/vdnma_gym:Switch4-v0')
#q = torch.load('minimal-marl/qmix_Switch4-v0')
#q = torch.load('minimal-marl/vdnma_gym:TrafficJunction10-v0')
#q = torch.load('minimal-marl/qmix_TrafficJunction10-v0')
#q = torch.load('minimal-marl/qmix_TrafficJunction4-v0')
score = 0
action_num = env.action_space
sampling_num = 10000
alpha=0.05
q=q.cuda()
radius =10000
R=0
alpha=0.001
done = [False for _ in range(env.n_agents)]
state = env.reset()
Rtot=10000
action_tot=[]
reward_n = 0
print(env.action_space[0].n)
epsilon=0.016
calculate_if=True
attack=False
sigma_list=[0.1]
#sigma_list=[0.05,0.1,0.2,0.3,0.4,0.5]
for sigma in sigma_list:
    reward_tot=[]
    for i in range(episodes):
        reward_n = 0
        state = env.reset()
        done = [False for _ in range(env.n_agents)]
        state_n.append(state)
        i=0
        action_box =[]
        action_box_1 =[]
        action_box_2 =[]
        hidden_box = []
        radius1=[]
        radius2=[]
        agent1_reward=[]
        agent2_reward=[]
        p1=[]
        p2=[]
        if_list={}
        rad=[]
        action_tot=[]
        with torch.no_grad():
            hidden = q.init_hidden()
            fail=0
            radius={}
            pv={}
            state=torch.Tensor(state).cuda()
            #noise=[ torch.FloatTensor(*(state.shape)).normal_(0, sigma).cuda()
                            #for _ in tqdm(range(episodes)) ]
            while not all(done):   
                state=torch.Tensor(state).cuda()
                #state=state+torch.FloatTensor(*(state.shape)).normal_(0, sigma).cuda()
                #action, hidden = q.sample_action(state.unsqueeze(0), hidden, epsilon=0)
                if attack == True:
                    state_at = attacks.attack(q, state, sigma,env.n_agents,hidden,epsilon,noise)[0]
                    action,action_box,hidden_box= sample_action(state_at, hidden)
                else:
                    #state=torch.Tensor(state).cuda()
                    action,action_box,hidden_box= sample_action(state, hidden)
                #print('this step action is',action)
                #action= [[max(action_box_1,key = action_box_1.count),max(action_box_2,key = action_box_2.count)]]
                
                action_count={}
                for j in range(env.n_agents):
                    h= np.array(action_box[j]).tolist()
                    for i in range(env.action_space[0].n):
                        action_count.setdefault(j,[]).append(h.count(i))
                hidden=[]
                for i in action_count:
                    pv1=stats.binom_test(max(action_count[i]), n=sum(sorted(action_count[i])[-2:]), p=0.5)
                    index=action_box[i].index(max(action_box[i],key = action_box[i].count))
                    hidden.append(hidden_box[index][0,i,:])
                    agent_pro=confidence_bound(action_count[i],0.001)
                    radiusc = sigma/2 * (norm.ppf(max(agent_pro[:,0]))-norm.ppf(sorted(agent_pro[:,1])[-2]))
                    radius.setdefault(i,[]).append(radiusc)
                    pv.setdefault(i,[]).append(pv1)     
                #if i == 0:
                snapshot = env.get_snapshot()
                hidden =torch.Tensor(np.array(hidden)).unsqueeze(0)
                next_state, reward, done, info = env.step(action)
                #next_state, reward, done, info = env.step(action[0])
                if calculate_if ==True:
                    for i in range(env.n_agents):
                        Q_target=[]
                        for j in range(env.action_space[0].n):
                            env2.load_snapshot(snapshot)
                            action_new=action
                            action_new[i]=j
                            _,rew_j,_,_= env2.step(action_new)
                            Q_target.append(sum(rew_j))
                        otp=normalize(np.array(action_count[i]).reshape(1,-1),'max')
                        baseline = np.sum(otp* Q_target)
                        ifl=max(sum(reward) - baseline,0)
                        if_list.setdefault(i,[]).append(ifl)
                score += sum(reward)
                reward_n +=sum(reward)
                print(reward_n)
                state = next_state
                i=i+1
            print(fail)
            print(reward_n)
            reward_tot.append(reward_n)
    torch.save(radius,'radius_'+'vdn_'+ str(env_name)+str(sampling_num)+'_'+str(sigma) + '.pth')
    torch.save(pv,'pv_'+ 'vdn_'+str(env_name)+str(sampling_num)+'_'+str(sigma) + '.pth')
    torch.save(if_list,'vdn_rank'+'_'+str(sampling_num)+str(env_name)+'_'+str(sigma)+'.pth')
env.close()






