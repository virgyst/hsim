# -*- coding: utf-8 -*-
from sys import path
path.append('../')
import hsim.core.pymulate as pym
from hsim.core.chfsm import CHFSM, Transition, State
import pandas as pd
import numpy as np
from simpy import AnyOf
from copy import deepcopy
from random import choices,seed,normalvariate, expovariate
from hsim.core.stores import Store, Box       
from scipy import stats
import dill
import hsim.core.utils as utils
pd.set_option('future.no_silent_downcasting', True)
import copy
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque 
import math
import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_service_times(service_times_log):
    df = pd.DataFrame(service_times_log)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
        nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )
    
    def forward(self, x):
        return self.network(x)

import random
from collections import deque

class DQNAgent:
    def __init__(self, state_dim, action_dim,max_time):
        self.state_dim = state_dim
        self.dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon =1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.5
        self.model = DQN(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss() 
        self.accumulated_rewards = list([])#deque(maxlen=10000)
        self.current_time=0
        self.max_time= max_time
        self.episode_count = 0
        self.previous_total_reward = float('inf')
        self.rewards_window = deque(maxlen=100)
        self.sequence_length = action_dim
        self.current_sequence = []
        
    def save_model(self):
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'memory': list(self.memory),
                'previous_total_reward': self.previous_total_reward # Salva la memoria come lista
            }, 'dqn_model.pth')
            print("Model saved to dqn_model.pth")
    
    def load_model(self):
        try:
            checkpoint = torch.load('dqn_model.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            # print(f'questa è epsilon {self.epsilon}')
            self.memory = deque(checkpoint['memory'], maxlen=10000)
            self.previous_total_reward = checkpoint.get('previous_total_reward',float('inf') )  # Carica la memoria
            self.model.train()  # Imposta il modello in modalità di addestramento
            #self.model.eval()  # Imposta il modello in modalità di valutazione
            print("Model loaded from dqn_model.pth")
        except FileNotFoundError:
            print("No saved model found, starting with a new model")
    
    def update_last_transition(self):
        if not self.memory:
            print("Memory is empty, no transition to update.")
            return
        final_reward = self.calculate_final_reward()
        last_transition = self.memory[-1]
        last_transition = list(last_transition)
        last_transition[2]=final_reward
        last_transition[4]=True
        last_transition=tuple(last_transition)
        self.memory[-1]=last_transition
        self.accumulated_rewards.clear()
        self.save_model()
    def isdone(self,env):
        if env.now >= self.max_time:
            print("Simulation finished")
            # self.save_model()
            return True
        return False
            
    def remember(self, state, action, reward, next_state, done):
        state = np.array(state)
        next_state = np.array(next_state)
        state = np.nan_to_num(state, nan=9999)
        next_state = np.nan_to_num(next_state, nan=9999)
        if state is None or action is None or next_state is None:
            print("Invalid entry detected, skipping insertion")
            return
        self.accumulated_rewards.append(reward)
        transition = (state, action, reward, next_state, done)
        self.current_sequence.append(transition)
        if len(self.current_sequence) == self.sequence_length:
            self.memory.append(list(self.current_sequence))
            self.current_sequence = []
   
    class ReplayBuffer:
        def __init__(self, capacity, sequence_length):
            self.capacity = capacity
            self.memory = deque(maxlen=capacity)
            self.sequence_length = sequence_length

        def push(self, sequence):
            """Save a sequence of transitions"""
            self.memory.append(sequence)

        def sample(self, batch_size):
            """Sample a batch of sequences"""
            sequences = random.sample(self.memory, batch_size)
            return sequences

        def __len__(self):
            return len(self.memory)
        
    def act(self, state, type_col_index, entity_col_index):
        if isinstance(state, np.ndarray):
            state = np.nan_to_num(state, nan=9999)
            if np.random.rand() <= self.epsilon:
                # Filtra le righe dove 'type' è 0
                filtered_state = state[state[:, type_col_index] == 0]
                # print(f'filtered_state shape {filtered_state.shape}')
                if filtered_state.size > 0:
                    selected_value=random.choice(filtered_state[:, entity_col_index].tolist())
                    for i in range(len(filtered_state)):
                        if state[i][entity_col_index] == selected_value:
                            # print(i)
                            return i
                else:
                    i=-1
                    return i
            else:
                # Usa il modello per predire i Q-values e seleziona l'azione con il valore massimo
                # print("EXPLOIT")
                # print(state.shape)
                state_tensor = torch.FloatTensor(state)
                q_values = self.model(state_tensor)
                max_index=int(np.argmax(q_values.detach().numpy()[0]))
                # print(f"Max index: {max_index}")
                if max_index < 0 or max_index > len(state):
                    i=-1
                    return i
    
                return max_index
        else:
            print("State is not a NumPy array")
            return None
        
    
    def get_reward(self, state,colonne,action,flag,done):
                # Verifica che state sia un array NumPy
        if not isinstance(state, np.ndarray):
            raise ValueError("state deve essere un array NumPy")

        # Verifica che columns sia un array NumPy
        if not isinstance(colonne, np.ndarray):
            raise ValueError("columns deve essere un array NumPy")
        if action is None and flag==0:
            reward= -10
        if action is None and flag==1:
            reward=0
        if action is not None:
            reward =0
        if done:
            try:
                time_out_index = np.where(colonne == 'timeOut')[0][0]
                time_in_index = np.where(colonne == 'timeIn')[0][0]
            except IndexError:
                raise ValueError("columns deve contenere 'timeOut' e 'timeIn'")
            state[:, time_in_index] = np.nan_to_num(state[:, time_in_index], nan=9999)
            
            state[:, time_out_index] = np.nan_to_num(state[:, time_out_index], nan=9999)
            
            max_time_out = np.nanmax(state[:, time_out_index])
            min_time_in = np.nanmin(state[:, time_in_index])
            # makespan = np.nanmax(state[:, time_out_index]) - np.nanmin(state[:, time_in_index])
            if max_time_out == 9999:
                valid_times_out = state[:, time_out_index][state[:, time_out_index] != 9999]
                if len(valid_times_out) > 0:
                    max_time_out = np.nanmax(valid_times_out)
                    # print(f"Max time out: {max_time_out}")
                else:
                    # print("non ci sono tempi diversi")
                    max_time_out = 0
            # print(max_time_out)
            makespan = max_time_out - min_time_in
            # print(f"Makespan: {makespan}")
            if makespan == 0:
                reward = 0
                # print("Makespan is 0")
            else:
                reward = - makespan
            return reward
    def calculate_final_reward(self):
        final_reward = self.accumulated_rewards[-1]
        if lab.gate.Store.items:
            print("Store is not empty, penalizing")
            penalty = 500
            final_reward -= penalty
        return final_reward

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 
        minibatch = self.memory.sample(batch_size)
        
        for sequence in minibatch:
            states, actions, rewards, next_states, dones = zip(*sequence)
            
            # Converti le sequenze in tensori PyTorch
            states = torch.FloatTensor(np.nan_to_num(states, nan=9999))
            next_states = torch.FloatTensor(np.nan_to_num(next_states, nan=9999))
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones)
            
            # Calcola i target per ogni passo nella sequenza
            targets = rewards.clone()
            non_final_mask = (dones == 0)
            non_final_next_states = next_states[non_final_mask]
            
            if len(non_final_next_states) > 0:
                next_q_values = self.model(non_final_next_states).max(1)[0].detach()
                targets[non_final_mask] += self.gamma * next_q_values
            
            # Propaga il reward finale indietro attraverso la sequenza con sconto
            final_reward = rewards[-1]  # Reward finale dell'episodio
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    targets[t] = final_reward
                else:
                    targets[t] += final_reward * (self.gamma ** (len(rewards) - 1 - t))
            
            # Calcola i Q-values attuali
            q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Ottimizza il modello
            self.optimizer.zero_grad()
            loss = self.criterion(q_values, targets)
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
                    
    
class Entity:
    _id_counter=0
    def __init__(self,ID=None):
        if ID is None:
            self.ID = Entity._id_counter
            Entity._id_counter += 1
            # print(f"sto creando un Entity ID: {self.ID}")
        else:
            self.ID = ID
        self.rework = False
        self.serviceTime = dict()
        # self.pt['M3'] = 1

    def reset_id_counter():
        Entity._id_counter = 0
    @property
    def require_robot(self):
        if self.serviceTime['robot']>0:
            return True
        else:
            return False
    @property
    def ok(self):
        return not (self.rework and self.require_robot)
    def done(self):
        self.rework = False
                
class LabServer(pym.Server):
    def __init__(self,env,name=None,serviceTime=None,serviceTimeFunction=None):
        self.controller = None
        # serviceTime = 10
        super().__init__(env,name,serviceTime,serviceTimeFunction)
    def calculateServiceTime(self,entity=None,attribute='serviceTime'):
        if not entity.ok:
            return 20 #3.5 ### qua andrà messo 10/20 volte maggiore degli altri processing time?
        else:
            print("service time")
            print(super().calculateServiceTime(entity,attribute))
            return super().calculateServiceTime(entity,attribute)
    def completed(self):
        if self.var.entity.ok:
            self.controller.Messages.put(self.name)
    T2=Transition(pym.Server.Working, pym.Server.Blocking, lambda self: self.env.timeout(self.calculateServiceTime(self.var.entity)), action = lambda self: self.completed())
    T3=Transition(pym.Server.Blocking, pym.Server.Starving, lambda self: self.Next.put(self.var.entity),action=lambda self: [self.var.request.confirm(), self.sm.var.entity.done() if self.sm._name=='robot' else None])

def plot_service_times(service_times_log):
    df = pd.DataFrame(service_times_log)
    df.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.xlabel('Entity')
    plt.ylabel('Service Time')
    plt.title('Service Times of Entities')
    plt.legend(title='Service Stages')
    plt.show()


class Terminator(pym.Terminator):
    def __init__(self, env, capacity=np.inf):
        super().__init__(env, capacity)
        self.controller = None
        self.register = list()
    def completed(self):
        if not self.trigger.triggered:
            self.trigger.succeed()
    def put(self,item):
        self.register.append(self._env.now)
        self.controller.Messages.put('terminator')
        return super().put(item)
    def subscribe(self,item):
        self.register.append(self._env.now)
        self.controller.Messages.put('terminator')
        return super().subscribe(item)
class Gate(CHFSM):
    def __init__(self,env,statedim,actiondim,max_time):
        self.real = True
        self.default_request_count = 0
        self.capacity = 30 #check se è giusto 40 o 30 come era prima
        self.lab = None
        self.initialWIP = 12
        self.targetWIP = 12
        self.request = None
        self.message = env.event()
        self.WIP = 0
        self.WIPlist = list()
        self.dones=False 
        self.training_step=0
        self.dqn_agent = DQNAgent(state_dim=statedim, action_dim=actiondim,max_time=max_time)
        self.dqn_agent.load_model()
        # self.count=len(b)
        super().__init__(env) #prima era in fondo ai self
# # #AGGIUNTA
    def concatenate_state_dataframes(self, batch_df, machine_df):
        batch_df['type'] = 0  # 0 per batch
        machine_df['type'] = 1  # 1 per macchina
        dfi= pd.concat([batch_df, machine_df], axis=0, sort=False).reset_index(drop=True)
        dfi.to_excel('dfi.xlsx')
        return dfi

    def convertdataframe(self,df,flag=True):
        mapping = {
            "front": 1,
            "drill": 2,
            "robot": 3,
            "camera": 4,
            "back": 5,
            "press": 6,
            "manual": 7
        }
        df["ResourceName"] = df["ResourceName"].map(mapping)

        mapping2 = {
            "Working": 1,
            "Blocking": 2,
            "Starving": 3
        }  
        df["StateName"] = df["StateName"].map(mapping2)
        df["Entity"] = df["id"]
        
        # filtered_df = df[df["type"] == 0]

        # # Verifica e conversione della colonna "id" se è di tipo float
        # if filtered_df["id"].dtype == float:
        #     df.loc[df["type"] == 0, "id"] = df.loc[df["type"] == 0, "id"].astype(int)

        # # Verifica e conversione della colonna "Entity" se è di tipo float
        # if filtered_df["Entity"].dtype == float:
        #     df.loc[df["type"] == 0, "Entity"] = df.loc[df["type"] == 0, "Entity"].astype(int)
        
        df.to_excel('conversionenumpy.xlsx')
        data_array=df.to_numpy()
        column_names = df.columns.to_numpy()

        # print(column_names)
        if flag:
            return data_array
        else:
            return data_array, column_names

    
    
    def filtro(self,val):
        return lambda item: item == val
    
    def indexcolumns(self,column):
        type_col_index = list(column).index('type')
        entity_col_index = list(column).index('Entity')
        return type_col_index, entity_col_index
    
    def action_to_request(self,indice,current_state):
        if indice is None or indice < 0 or indice >= current_state.shape[0] :
            # print(f"Indice negativo: {indice}")
            return None
        element=None
        for i in range(current_state.shape[0]):
            if i == indice:
                element=current_state[i, 0]
                # print(f"Elemento selezionato: {element}")
                break
                
        if element is not None:
            row = lab.b_df.loc[lab.b_df['id'] == element]
            if not row.empty:
                values = row['Entity'].values[0]
                return values   
            else:
        # print(f"No entity found with id {element}")
                return None
        else:
            return None


    def get_state(self,b_df,val=True):
        machine_df=pd.DataFrame(self.env.log,columns=["ResourceName","StateName","timeIn","timeOut"])
        machine_df=machine_df.loc[machine_df.ResourceName.isin(["front","drill","robot","camera","back","press","manual"])]
        batch_df = b_df[b_df['Entity'].isin(lab.gate.Store.items)] # Filtra solo le righe del batch che sono presenti nello store
        spazio_stati=self.concatenate_state_dataframes(batch_df,machine_df)
        spazio_stati.to_excel('spazio_stati.xlsx')
        spazio_stati,index = self.convertdataframe(spazio_stati,flag=False)
        # print(spazio_stati.dtype)
        if val:
            return spazio_stati
        else:
            return spazio_stati, index


#FINE AGGIUNTA
    def build(self):
        self.Store = pym.Store(self.env,self.capacity)
        self.Messages = pym.Store(self.env)
    def put(self,item):
        return self.Store.put(item)
    class Loading(State):
        def _do(self):
            # print('Load: %d' %self.sm.initialWIP)
            self.sm.initialWIP -= 1
            self.fw()
    class Waiting(State):
        initial_state = True
        def _do(self):
            self.sm.message = self.Messages.subscribe()
            if self.sm.initialWIP > 0:
                self.initial_timeout = self.env.timeout(1)
            else:
                self.initial_timeout = self.env.event()
    class Forwarding(State):
        def _do(self):
            self.message.confirm()
            if self.message.value == 'terminator':
                self.sm.WIP -= 1
                self.sm.WIPlist.append([self.env.now,self.WIP])
                # print(self.sm.WIPlist)
            self.FIFO()
            self.CONWIP()
    
    def CONWIP(self):
        if self.message.value == 'terminator':
            self.fw()
    def FIFO(self):
        pass
  
    def fw(self):    
        if self.request is None:
            flag = 0
            current_state, columns = self.get_state(lab.b_df, val=False)
            type_col_index, entity_col_index = self.indexcolumns(columns)
            action = self.dqn_agent.act(current_state, type_col_index, entity_col_index)  # index intero
            actionselct = self.action_to_request(action, current_state)  # entity

            if not self.Store:
                flag = 1
            if actionselct is not None:
                self.request = self.Store.get(self.filtro(actionselct))
                self.Next.put(self.request.value)
                self.request = None
                self.WIP += 1
                self.WIPlist.append([self.env.now, self.WIP])

            next_state = self.get_state(lab.b_df)
            reward = self.dqn_agent.get_reward(current_state, columns, actionselct, flag)
            self.dqn_agent.remember(current_state, action, reward, next_state, self.dones)

            if len(self.dqn_agent.memory) > 150:
                self.dqn_agent.replay(30)
        else:
            pass
    T0 = Transition(Waiting,Loading,lambda self: self.initial_timeout)
    T1 = Transition(Waiting,Forwarding,lambda self: self.sm.message)
    T2 = Transition(Loading,Waiting,None)
    T3 = Transition(Forwarding,Waiting,None)
         

class Router(pym.Router):
    def __deepcopy(self,memo):
        super().deepcopy(self,memo)
    def __init__(self, env, name=None):
        super().__init__(env, name)
        self.var.requestOut = []
        self.var.sent = []
        self.putEvent = env.event()
    def build(self):
        self.Queue = Box(self.env)
    def condition_check(self,item,target):
        return True
    def put(self,item):
        if self.putEvent.triggered:
            self.putEvent.restart()
        self.putEvent.succeed()
        return self.Queue.put(item)
    class Sending(State):
        initial_state = True
        def _do(self):
            self.sm.putEvent.restart()
            self.sm.var.requestIn = self.sm.putEvent
            self.sm.var.requestOut = [item for sublist in [[next.subscribe(item) for next in self.sm.Next if self.sm.condition_check(item,next)] for item in self.sm.Queue.items] for item in sublist]
            if self.sm.var.requestOut == []:
                self.sm.var.requestOut.append(self.sm.var.requestIn)
    S2S2 = Transition(Sending,Sending,lambda self:AnyOf(self.env,self.var.requestOut),condition=lambda self:self.var.requestOut != [])
    def action2(self):
        self.Queue._trigger_put(self.env.event())
        if not hasattr(self.var.requestOut[0],'item'):
            return
        for request in self.var.requestOut:
            if not request.item in self.Queue.items:
                request.cancel()
                continue
            if request.triggered:
                if request.check():
                    request.confirm()
                    self.Queue.forward(request.item)
                    continue
    S2S2._action = action2
'''  
from pymulate import RouterNew
class Router(RouterNew):
    def __init__(self, env, name=None):
        capacity=1
        super().__init__(env, name,capacity)
'''
class RobotSwitch1(Router):
    def condition_check(self, item, target):
        if item.require_robot:
            item.rework = True
        if item.require_robot and target.name == 'convRobot1S':
            return True
        elif not item.require_robot and target.name != 'convRobot1S':
            return True
        else:
            return False
            
class RobotSwitch2(Router):
    def condition_check(self, item, target):
        if len(target.Next)<2:
            item.rework = False
            return True
        else:
            item.rework = True
            return False    

class CloseOutSwitch(Router):
    def condition_check(self, item, target):
        if item.ok and type(target) == Terminator:
            return True
        elif not item.ok and type(target) != Terminator:
            return True
        else:
            return False
        
# class Conveyor(pym.ParallelServer):
#     def __init__(self,env,name=None,serviceTime=None,serviceTimeFunction=None,capacity=1):
#         self._capacity = capacity
#         serviceTime = capacity*3.5
#         super().__init__(env,name,serviceTime,serviceTimeFunction,capacity)
class Conveyor(pym.Conveyor):
    def __init__(self,env,name=None,capacity=3):
        super().__init__(env,name,capacity,0.75)
        
def newDT():
    lab = globals()['lab']
    deepcopy(lab)

def newEntity(): #viene usata all interno di batch create
    e = Entity()
    e.serviceTime['front'] = 10.52
    e.serviceTime['drill'] = 40 #choices([3.5, 8.45, 9.65, 11.94],weights=[5,30,30,35])[0]
    e.serviceTime['robot'] = choices([0, 81, 105, 108 ,120],weights=[91,3,2,2,2])[0]
    e.serviceTime['camera'] = 3.5+expovariate(1/7.1)
    e.serviceTime['back'] = choices([3.5,10.57],weights=[0.1,0.9])[0]
    if e.serviceTime['back']>0:
        e.serviceTime['press'] = 3.5+expovariate(1/9.5)
    else:
        e.serviceTime['press'] = 3.5
    e.serviceTime['manual'] = max(np.random.normal(9.2,1),0)
    return e


def batchCreate(seed=1,numJobs=10,return_both=False):
    np.random.seed(seed)
    jList = []
    complist = []
    while len(jList)<numJobs:
        e=newEntity()
        # num = round(np.random.triangular(1,numJobs/2,numJobs))
        # # print(num)
        # for i in range(num):
        jList.append(e)
        entity_info = {
                'id': e.ID,
                'Entity': e,
                'serviceTime': e.serviceTime
                }
        complist.append(entity_info)
        if len(jList)>=numJobs:  # l'ho aggiunto io per evitare che si creino più entità di quelle richieste
            break
    if return_both:
        return jList, complist
    else:
        return jList
    

class Lab():
    def __init__(self,b,maxtime):
        conveyTime = 6
        self.env = pym.Environment() #crea l'ambiente
        # self.g = Generator(self.env) #genera un nuovo pezzo
        
        self.a=self.env.log  #ambiente stati macchine è un dataframe della lista self.state_log
        self.machine_df=pd.DataFrame(self.a,columns=["ResourceName","StateName","timeIn","timeOut"])
        self.machine_df=self.machine_df.loc[self.machine_df.ResourceName.isin(["front","drill","robot","camera","back","press","manual"])]
        self.machine_df.to_excel('machine_df.xlsx')
        
        b_df = pd.DataFrame(b)
        self.b_df=pd.concat([b_df['id'],b_df['Entity'], pd.DataFrame(b_df['serviceTime'].tolist())],axis=1)
        self.b_df.to_excel('b_df.xlsx')
        self.dim_state=self.b_df.shape[1]+self.machine_df.shape[1]+1 #14
        self.dim_action=len(b_df) #come la capacity
        self.list = []
        self.gate = Gate(self.env,statedim=self.dim_state,actiondim=self.dim_action,max_time=maxtime) #crea il gate

        # DR= despaching rule OR=order release 
        # self.conv1 = Conveyor(self.env,capacity=3)
        self.conv1S = pym.Server(self.env,serviceTime=conveyTime) 
        self.conv1Q = pym.Queue(self.env,capacity=2)
        self.front = LabServer(self.env,'front')
        # self.conv2 = Conveyor(self.env,capacity=3)
        self.conv2S = pym.Server(self.env,serviceTime=conveyTime)
        self.conv2Q = pym.Queue(self.env,capacity=2)
        self.drill = LabServer(self.env,'drill')
        # self.conv3 = Conveyor(self.env,capacity=3)
        self.conv3S = pym.Server(self.env,serviceTime=conveyTime)
        self.conv3Q = pym.Queue(self.env,capacity=2)

        
        self.switch1 = RobotSwitch1(self.env)
        # self.convRobot1 = Conveyor(self.env,'convRobot1',capacity=3)
        self.convRobot1S = pym.Server(self.env,serviceTime=conveyTime,name='convRobot1S')
        self.convRobot1Q = pym.Queue(self.env,capacity=2)

        # self.bridge = Conveyor(self.env,capacity=3)
        self.bridgeS = pym.Server(self.env,serviceTime=conveyTime)
        self.bridgeQ = pym.Queue(self.env,capacity=2)

        # self.convRobot2 = Conveyor(self.env,'convRobot2',capacity=3)
        self.convRobot2S = pym.Server(self.env,serviceTime=conveyTime)
        self.convRobot2Q = pym.Queue(self.env,capacity=2)

        self.switch2 = RobotSwitch2(self.env)
        # self.convRobot3 = Conveyor(self.env,capacity=3)
        self.convRobot3S = pym.Server(self.env,serviceTime=conveyTime)
        self.convRobot3Q = pym.Queue(self.env,capacity=2)

        self.robot = LabServer(self.env,'robot')
        # self.convRobotOut = Conveyor(self.env,capacity=3)
        self.convRobotOutS = pym.Server(self.env,serviceTime=conveyTime)
        self.convRobotOutQ = pym.Queue(self.env,capacity=2)
        # self.conv5 = Conveyor(self.env,capacity=3)
        self.conv5S = pym.Server(self.env,serviceTime=conveyTime)
        self.conv5Q = pym.Queue(self.env,capacity=2)

        self.camera = LabServer(self.env,'camera')
        # self.conv6 = Conveyor(self.env,capacity=3)
        self.conv6S = pym.Server(self.env,serviceTime=conveyTime)
        self.conv6Q = pym.Queue(self.env,capacity=2)

        self.back = LabServer(self.env,'back')
        # self.conv7 = Conveyor(self.env,capacity=3)
        self.conv7S = pym.Server(self.env,serviceTime=conveyTime)
        self.conv7Q = pym.Queue(self.env,capacity=2)

        self.press = LabServer(self.env,'press')
        # self.conv8 = Conveyor(self.env,capacity=3)
        self.conv8S = pym.Server(self.env,serviceTime=conveyTime)
        self.conv8Q = pym.Queue(self.env,capacity=2)

        self.manual = LabServer(self.env,'manual')
        self.outSwitch = CloseOutSwitch(self.env)
        self.terminator = Terminator(self.env)
        
        # self.g.Next = self.gate
        self.gate.Next = self.conv1S
        
        # self.conv1.Next = self.front
        self.conv1S.Next = self.conv1Q
        self.conv1Q.Next = self.front

        self.front.Next = self.conv2S
        # self.conv2.Next = self.drill
        self.conv2S.Next = self.conv2Q
        self.conv2Q.Next = self.drill
        self.drill.Next = self.conv3S
        self.conv3S.Next = self.conv3Q
        self.conv3Q.Next = self.switch1
        # self.conv3.Next = self.switch1
        
        self.switch1.Next = [self.convRobot1S,self.bridgeS]
        self.convRobot1S.Next = self.convRobot1Q
        self.convRobot1Q.Next = self.switch2

        self.switch2.Next = [self.convRobot2S,self.convRobot3S]
        self.convRobot2S.Next = self.convRobot2Q
        self.convRobot2Q.Next = self.robot

        self.convRobot3S.Next = self.convRobot3Q
        self.convRobot3Q.Next = self.convRobotOutS

        self.robot.Next = self.convRobotOutS
        self.convRobotOutS.Next = self.convRobotOutQ

        self.convRobotOutQ.Next = self.conv5S
        self.bridgeS.Next = self.bridgeQ
        self.bridgeQ.Next = self.conv5S

        
        self.conv5S.Next = self.conv5Q
        self.conv5Q.Next = self.camera

        self.camera.Next = self.conv6S
        self.conv6S.Next = self.conv6Q
        self.conv6Q.Next = self.back

        self.back.Next = self.conv7S
        self.conv7S.Next = self.conv7Q
        self.conv7Q.Next = self.press

        self.press.Next = self.conv8S
        self.conv8S.Next = self.conv8Q
        self.conv8Q.Next = self.manual

        self.manual.Next = self.outSwitch
        self.outSwitch.Next = [self.conv1S,self.terminator]
        
        for x in [self.front,self.drill,self.robot,self.camera,self.back,self.press,self.manual]:
            x.controller = self.gate
        self.terminator.controller = self.gate
        self.gate.lab = self

       

    def run(self,Tend):
        self.env.run(Tend)
        if self.gate.dqn_agent.isdone(self.env):
            self.gate.dqn_agent.update_last_transition()
            print("Operazione completata")
        else:
            print("Operazione non completata")
        
       #return pd.DataFrame(self.env.state_log)
        return self.env.state_log


import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
Entity.reset_id_counter()

makespan=list(pd.read_excel('makespan.xlsx')['Makespan'])
for i in range(1):
    Entity.reset_id_counter()
    model_path = 'dqn_model.pth'
    if i == 0:
        if os.path.exists(model_path):
            print(f"Modello esistente trovato al ciclo {i}.")
            os.remove(model_path)
        else:
            print(f"Nessun modello esistente trovato al ciclo {i}. Inizio con un nuovo modello.")     #     
    s,b = batchCreate(0,numJobs=10,return_both=True)
    
    lab=Lab(b,900)
    # lab.gate.Store.items = [item['Entity'] for item in b]
    lab.gate.Store.items= copy.copy(s)
    if i > 0:
        assert lab.gate.dqn_agent.model is not None, "Model not loaded correctly"
    lab.run(900)

    df = pd.DataFrame(lab.env.state_log, columns=["Resource","ResourceName","State","StateName","Entity","?","timeIn","timeOut"])
    df= df.loc[df.ResourceName.isin(["front","drill","robot","camera","back","press","manual"])]
    mks=df.timeOut.max()-df.timeIn.min()

    makespan.append(mks)
    # Verifica che il modello sia stato salvato correttamente
    try:
        checkpoint = torch.load('dqn_model.pth')
        assert 'model_state_dict' in checkpoint, "Model state dict not found in checkpoint"
        print("Model saved correctly")
    except FileNotFoundError:
        print("Model not saved correctly")


file_path = 'makespan.xlsx'
if os.path.exists(file_path):
    os.remove(file_path)

mksdf = pd.DataFrame(makespan, columns=['Makespan'])
mksdf.to_excel('makespan.xlsx')
print(makespan)
# Crea il grafico a linee
plt.figure(figsize=(10, 6))
plt.plot(makespan, marker='o', linestyle='-', color='b', label='Makespan')

# Aggiungi etichette e titolo
plt.xlabel('Iterazione')
plt.ylabel('Makespan (secondi)')
plt.title('Variazione del Makespan nel Tempo')
plt.legend()

# Mostra il grafico
plt.grid(True)
plt.show()


