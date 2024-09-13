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

def funzioneconduereturn():
    ris1 = "primo risultato"
    ris2 = "secondo risultato"
    return ris1, ris2

def create_entities_dataframe(generator):
    data = []
    for entity in generator.entities:
        row = {'ID': entity.ID, **entity.serviceTime}
        data.append(row)
    df = pd.DataFrame(data)
    return df

def plot_service_times(service_times_log):
    # Convert the list of dictionaries to a DataFrame for easier plotting
    df = pd.DataFrame(service_times_log)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
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
        self.epsilon_min = 0.01
        self.model = DQN(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss() 
        self.accumulated_rewards = deque(maxlen=10000)
        self.current_time=0
        self.max_time= max_time
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if not done:
            if len(self.accumulated_rewards) > 0:
                self.accumulated_rewards[-1] += reward
        else:
            delayed_reward = self.get_reward(state)
            self.accumulated_rewards.append(delayed_reward)
        # print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}")
        # print(f"Memory before append: {self.memory}")
        # self.memory.append((state, action, reward, next_state, done))
        # print(f"Memory after append: {self.memory}")
        
        # if not done:
        #     print(f"Accumulated rewards before: {self.accumulated_rewards}")
        #     if len(self.accumulated_rewards) > 0:
        #         self.accumulated_rewards[-1] += reward
        #     print(f"Accumulated rewards after: {self.accumulated_rewards}")
        # else:
        #     delayed_reward = self.get_reward(state)
        #     print(f"Delayed reward: {delayed_reward}")
        #     self.accumulated_rewards.append(delayed_reward)
        #     print(f"Accumulated rewards after append: {self.accumulated_rewards}")
    

    # def act(self, state): 
    #     if isinstance(state, pd.DataFrame):
    #         # state = state.fillna()  # Riempie i valori NaN con 0
    #         state = state.infer_objects(copy=False)
    #         type_col_index = state.columns.get_loc('type')
    #         entity_index= state.columns.get_loc('Entity')
    #         states = state.to_numpy()
    #         if np.random.rand() <= self.epsilon:
    #             filtered_state = states[state['type'] == 0]
    #             ris=random.choice(filtered_state['Entity'].tolist())
    #         else:
    #             return print("ERROR")
    #     q_values= self.model.predict(state)
    #     return np.argmax(q_values[0])

    def act(self, state):
        if isinstance(state, pd.DataFrame):
            # Converti il DataFrame in un array numpy
            state = state.infer_objects(copy=False)
            states = state.to_numpy()
            
            if np.random.rand() <= self.epsilon:
                # Filtra le righe dove 'type' è 0
                type_col_index = state.columns.get_loc('type')
                entity_col_index = state.columns.get_loc('Entity')
                filtered_state = states[states[:, type_col_index] == 0]
                
                if filtered_state.size > 0:
                    # Seleziona casualmente un valore dalla colonna 'Entity'
                    return random.choice(filtered_state[:, entity_col_index].tolist())
                else:
                    print("No entities with type 0 found")
                    return None
            else:
                # Usa il modello per predire i Q-values e seleziona l'azione con il valore massimo
                q_values = self.model.predict(states)
                return np.argmax(q_values[0])
        else:
            print("State is not a DataFrame")
            return None
        
    def update_epsilon(self):
        # Aggiorna epsilon per ridurre gradualmente l'esplorazione
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay      
    
    def get_reward(self, state):
        makespan = state['timeOut'].max() - state['timeIn'].min()
        reward = self.calculate_reward(makespan)
        return reward

    def concatenate_state_dataframes(self, batch_df, machine_df):
        batch_df['type'] = 0  # 0 per batch
        machine_df['type'] = 1  # 1 per macchina
        df= pd.concat([batch_df, machine_df], axis=0, sort=False).reset_index(drop=True)
        df.to_excel('df.xlsx')
        return df
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if isinstance(state, pd.DataFrame):
                state=state.fillna(-1)
                state = state.to_numpy()
            if isinstance(next_state, pd.DataFrame):
                next_state =next_state.fillina(-1)
                next_state = next_state.to_numpy()
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.model(next_state)).item()
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def isdone(self,env):
        if env.now >= self.max_time:
            print("Simulation finished")
            final_reward = self.calculate_final_reward()
            print(f"Final Reward: {final_reward}")
            return True
        return False
    
       
    def calculate_final_reward(self):
        if len(self.accumulated_rewards) > 0:
            final_reward = sum(self.accumulated_rewards)
            return final_reward
        return 0
    
    def calculate_reward(self, makespan):
        return -makespan
class Generator(pym.Generator):
    def __init__(self,env,name=None,serviceTime=2,serviceTimeFunction=None):
        super().__init__(env,name,serviceTime,serviceTimeFunction)
        self.count = 0
        self.entities=pd.DataFrame(columns=['id', 'Entity', 'front', 'drill', 'robot', 'camera', 'back', 'press', 'manual'])
    def createEntity(self):
        self.count += 1
        # return Entity()
        e = Entity()#ID=self.count)
        #e.serviceTime = dict()
        e.serviceTime['front'] = 10.52
        e.serviceTime['drill'] = choices([3.5, 8.45, 9.65, 11.94],weights=[5,30,30,35])[0] ## Insert the possibility of skipping this stage
        e.serviceTime['robot'] = choices([0, 81, 105, 108 ,120],weights=[91,3,2,2,2])[0]
        # e.serviceTime['camera'] = choices([3,9,12,18,24],weights=[2,3,1,2,2])[0]
        e.serviceTime['camera'] = 3.5+expovariate(1/7.1)
        e.serviceTime['back'] = choices([3.5,10.57],weights=[0.1,0.9])[0]
        # e.serviceTime['press'] = choices([3,9,15])[0]
        if e.serviceTime['back']>0:
            e.serviceTime['press'] = 3.5+expovariate(1/9.5)
        else:
            e.serviceTime['press'] = 3.5
        e.serviceTime['manual'] = max(np.random.normal(9.2,1),0)
        #print(f"Entity {self.count} service times: {e.serviceTime}")
        # self.entities.append(e)
        service_time_dict = {f'serviceTime_{i}': [time] for i, time in enumerate(e.serviceTime)}
        entity_dict = {
            'id': e.ID,
            'Entity': e,
            'front': e.serviceTime['front'],
            'drill': e.serviceTime['drill'],
            'robot': e.serviceTime['robot'],
            'camera': e.serviceTime['camera'],
            'back': e.serviceTime['back'],
            'press': e.serviceTime['press'],
            'manual': e.serviceTime['manual']
        }

        entity_df = pd.DataFrame([entity_dict])
        lab.b_df = pd.concat([lab.b_df, entity_df], ignore_index=True)
        # print(f"ho aggiunto un'unità al batch:\n{lab.b_df.iloc[-1]}")
        # print("il nuovo batch è:")
        # print(lab.b_df)
        # print(lab.b_df.shape[1])
        # self.entities = pd.concat([self.entities, entity_df], ignore_index=True)
        # entity_df = pd.DataFrame([e.ID, e, pd.DataFrame(e.serviceTime)], columns=['id', 'Entity', 'front', 'drill', 'robot', 'camera', 'back', 'press', 'manual'])
        # self.entities = pd.concat([self.entities, entity_df], ignore_index=True)
        return e
    # def get_entities_service_times(self,entities):
    # # Convert the 'serviceTime' column to a DataFrame
    #     service_times_df = pd.DataFrame(entities['serviceTime'].tolist())
    
    # # Concatenate the 'id', 'Entity', and the new service times DataFrame
    #     data = pd.concat([entities[['id', 'Entity']], service_times_df], axis=1)
    #     print(data)
    #     return data
    
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
    def __init__(self,env,statedim,actiondim):
        self.real = True
        self.default_request_count = 0
        self.capacity = 30
        self.lab = None
        self.initialWIP = 12
        self.targetWIP = 12
        self.request = None
        self.message = env.event()
        self.WIP = 0
        self.WIPlist = list()
        self.dones=False 
        self.dqn_agent = DQNAgent(state_dim=statedim, action_dim=actiondim,max_time=3600)
        super().__init__(env) #prima era in fondo ai self
# # #AGGIUNTA
    def filtro(self,val):
        # if val in lab.gate.Store.items:
        #     return True
        # print("no filtro")
        # return False
        return lambda item: item == val

    
    def get_current_state(self,b_df):
        machine_df=pd.DataFrame(self.env.log,columns=["Resource","ResourceName","State","StateName","Entity","Store","timeIn","timeOut"])
        machine_df=machine_df.loc[machine_df.ResourceName.isin(["front","drill","robot","camera","back","press","manual"])]
        batch_df = b_df[b_df['Entity'].isin(lab.gate.Store.items)] # Filtra solo le righe del batch che sono presenti nello store
        self.spazio_staticor=self.dqn_agent.concatenate_state_dataframes(batch_df,machine_df)
        # self.spazio_staticor.to_excel('spazio_stati.xlsx')
        self.spazio_staticor=self.spazio_staticor.fillna(-1)
        return self.spazio_staticor
    
    def action_to_request(self,state):
        element=self.dqn_agent.act(state)
        if self.filtro(element):
            return element
        else:
            print("Errore")
            return None 

    def get_next_state(self,b):
        machine_df = pd.DataFrame(self.env.log, columns=["Resource", "ResourceName", "State", "StateName", "Entity", "Store", "timeIn", "timeOut"])
        machine_df=machine_df.loc[machine_df.ResourceName.isin(["front","drill","robot","camera","back","press","manual"])]
        batch_df = b[b['Entity'].isin(lab.gate.Store.items)]  # Filter only the rows of the batch that are present in the store
        self.spazio_statifu = self.dqn_agent.concatenate_state_dataframes(batch_df, machine_df)
        self.spazio_statifu = self.spazio_statifu.fillna(-1)
        return self.spazio_statifu
    # def update_delayed_reward(self, reward):
    #     if len(self.accumulated_rewards) > 0:
    #         self.accumulated_rewards[-1] += reward # If there are, add the new reward to the last element
    #     else:
    #         self.accumulated_rewards.append(reward) 

    # def doneboole(self,env):
    #     self.dones=self.dqn_agent.is_done(env)
    #     return self.dones
    
  
    
    
    # def condition(self, item, scheduled_request):
    #     if scheduled_request in self.Store.items:
    #         # return item == self.Store.items[scheduled_request]
    #         return lab.gate.Store.get(scheduled_request)
    #     else:
    #         # Gestisci il caso in cui l'indice sia fuori dai limiti
    #         print(f"Indice {scheduled_request} fuori dai limiti")
    #         return False

    
    # def get(self, condition):
    #     for item in self.Store.items:
    #         if condition(item):
    #             return item
    #     return None

    # def custom_condition(self, item, scheduled_request):
    #     return self.condition(item, scheduled_request)

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
            self.FIFO()
            self.CONWIP()
    
    def CONWIP(self):
        if self.message.value == 'terminator':
            self.fw()
    def FIFO(self):
        pass
  
    def fw(self):    
        if self.request is None:
           #self.request = self.Store.get(self.filtro('<__main__.Entity object at 0x00000222C40317D0>'))
#INIZIO NUOVO CODICE
            current_state = self.get_current_state(lab.b_df)
            actionselct = self.action_to_request(current_state)
            
            reward = self.dqn_agent.get_reward(current_state)
            self.dones =self.dqn_agent.isdone(self.env)
            self.request = self.Store.get(self.filtro(actionselct))
        # else:
       #self.request = self.Store.get()
        #     print(type(self.request))
        #     self.default_request_count += 1
            # print("NON FUNZIONA")
            # Controllo se la soglia è stata superata
        # if len(self.dqn_agent.memory) > 30:
        #     # Perform replay to train the network
        #     self.dqn_agent.replay()
        #     self.default_request_count = 0            
#fine nuovo codice
        # try:
            
        self.Next.put(self.request.value)
        self.request = None
        self.WIP += 1
        self.WIPlist.append([self.env.now,self.WIP])
        next_state=self.get_next_state(lab.b_df)
        print("arrivoqua1")
        self.dqn_agent.remember(current_state, actionselct, reward, next_state, self.dones)
        print("arrivoqua2")
        if len(self.dqn_agent.memory) > 30:
            # Perform replay to train the network
            print(">30")
            self.dqn_agent.replay(30)
            print("ciaociao")
            self.default_request_count = 0    

        # except:
        #     print("non so")
        #     pass
        
            # print('Empty at %s' %self.env.now)
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
    def __init__(self,b):
        conveyTime = 6
        self.env = pym.Environment() #crea l'ambiente
        self.g = Generator(self.env) #genera un nuovo pezzo
        
        self.a=self.env.log  #ambiente stati macchine è un dataframe della lista self.state_log
        self.machine_df=pd.DataFrame(self.a,columns=["Resource","ResourceName","State","StateName","Entity","Store","timeIn","timeOut"])
        self.machine_df=self.machine_df.loc[self.machine_df.ResourceName.isin(["front","drill","robot","camera","back","press","manual"])]
        self.machine_df.to_excel('machine_df.xlsx')
        
        b_df = pd.DataFrame(b)
        self.b_df=pd.concat([b_df['id'],b_df['Entity'], pd.DataFrame(b_df['serviceTime'].tolist())],axis=1)
        self.dim_state=self.b_df.shape[1]+self.machine_df.shape[1]+1 #18 in totale 9 batch 8 macchine +1 
       
        self.dim_action=30 #come la capacity

        self.gate = Gate(self.env,statedim=self.dim_state,actiondim=self.dim_action) #crea il gate

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
        
        self.g.Next = self.gate
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
       #return pd.DataFrame(self.env.state_log)
        return self.env.state_log

import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
Entity.reset_id_counter()
s,b = batchCreate(0,numJobs=10,return_both=True) #batchedExp serve per inizializzare 
print(s)
lab=Lab(b)
lab.gate.Store.items=copy.copy(s)
#inserisco ipotetiche righe messe sotto il commento
lab.run(3600)
df = pd.DataFrame(lab.env.state_log, columns=["Resource","ResourceName","State","StateName","Entity","?","timeIn","timeOut"])
df= df.loc[df.ResourceName.isin(["front","drill","robot","camera","back","press","manual"])]
fig = utils.createGantt(df)
fig.show()