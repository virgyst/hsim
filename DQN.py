# -*- coding: utf-8 -*-
from sys import path
path.append('../')
import hsim.core.pymulate as pym
from hsim.core.chfsm import CHFSM, Transition, State
import pandas as pd
import numpy as np
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque 
import torch.nn.init as init


def init_weights_he(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.network.apply(init_weights_he)

    def forward(self, x):
        return self.network(x)
    
import random
from collections import deque

class DQNAgent:
    def __init__(self, To, state_dim, action_dim,max_time):
        self.state_dim = state_dim
        self.dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99 #0.99
        self.epsilon =1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.03 #0.03
        self.total_production_time = To
        self.model = DQN(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001) #0.0001
        self.criterion = nn.SmoothL1Loss()#nn.MSELoss() 
        self.accumulated_rewards = list([])#deque(maxlen=10000)
        self.current_time=0
        self.max_time= max_time
        self.episode_count = 0
        self.previous_total_reward = float('inf')
        self.rewards_window = deque(maxlen=100)
        self.finreward=0

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
        last_episode = [] 
        count = 0
        for transition in reversed(self.memory):
            count += 1
            if transition[4]:  # Se done è True
                break
            last_episode.insert(0, transition)
        
        if not last_episode:
            print("No complete episode found in memory.")
            return
        
        final_reward = self.calculate_final_reward()
        for t in range(len(last_episode)):
            state, action, reward, next_state, done = last_episode[t]
            discount_factor = self.gamma ** (len(last_episode) - t - 1)
            if t == len(last_episode) - 1: 
                last_episode[t] = (state, action, final_reward, next_state, True)
            else:
                discounted_reward=discount_factor * final_reward
                last_episode[t] = (state, action, discounted_reward, next_state, done)
        for _ in range(len(last_episode)):
            self.memory.pop()
        self.memory.extend(last_episode)
        self.accumulated_rewards.clear()
        self.save_model()
    
     
    def remember(self, state, action, reward, next_state, done):
        state = np.array(state)
        next_state = np.array(next_state)
        state = np.nan_to_num(state, nan=9999)
        next_state = np.nan_to_num(next_state, nan=9999)
        if state is None or action is None or reward is None or next_state is None:
            print("Invalid entry detected, skipping insertion")
        # if action != -1 or reward is not None:
        self.accumulated_rewards.append(reward)
        # print(f"Accumulated rewards: {reward} che aggiungo")
        self.memory.append((state, action, reward, next_state, done))
            # else:
            # pass


    def isdone(self,env):
        if env.now >= self.max_time:
            return True
        return False
       
    def calculate_final_reward(self):
        reward_f=self.finreward
        if lab.gate.Store.items:
            print("Store is not empty, penalizing")
            penalty = 500  
            reward_f -= penalty
        return reward_f
   

    def act(self, state, type_col_index, entity_col_index):
        if isinstance(state, np.ndarray):
            state = np.nan_to_num(state, nan=9999)
            if np.random.rand() <= self.epsilon:
                filtered_state = state[state[:, type_col_index] == 0]
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
                state_tensor = torch.FloatTensor(state)
                q_values = self.model(state_tensor)
                max_index=int(np.argmax(q_values.detach().numpy()[0]))
                if max_index < 0 or max_index > len(state):
                    i=-1
                    return i
    
                return max_index
        else:
            print("State is not a NumPy array")
            return None

    
    def get_reward(self, state,colonne,action,flag):
        if not isinstance(state, np.ndarray):
            raise ValueError("state deve essere un array NumPy")
        if not isinstance(colonne, np.ndarray):
            raise ValueError("columns deve essere un array NumPy")
        if action==-1 and lab.gate.Store.items:
            reward= -100
        else:
            if action ==-1:
                reward= -50
            else:
                reward=0
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
            else:
                max_time_out = 0
        makespan = max_time_out
        reward =makespan
        return reward

    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            print("Not enough samples in memory to replay.")
            return
        
        # Trova gli indici degli episodi completi
        episode_indices = []
        current_episode = []
        for idx, transition in enumerate(self.memory):
            current_episode.append(transition)
            if transition[4]:  # Se done è True
                episode_indices.append(current_episode)
                current_episode = []
        
        if len(episode_indices) < batch_size:
            # print("Not enough episodes in memory to replay.")
            return
        
        # Assegna priorità decrescente agli episodi in base alla loro posizione nella memoria
        priorities = np.linspace(1.0, 0.1, len(episode_indices))
        probabilities = priorities / priorities.sum()
        
        # Campiona batch_size episodi completi
        sampled_episodes = np.random.choice(len(episode_indices), batch_size,p=probabilities)
        minibatch = [episode_indices[idx] for idx in sampled_episodes]
        # minibatch = random.sample(episode_indices, batch_size)
        total_loss = 0.0  # Inizializza total_loss a zero
        num_transitions = 0  # Contatore per il numero di transizioni
        
        for episode in minibatch:
            for state, action, reward, next_state, done in episode:
                next_state = torch.FloatTensor(np.nan_to_num(next_state, nan=9999))
                state = torch.FloatTensor(np.nan_to_num(state, nan=9999))
                action = torch.LongTensor([action])
                reward = torch.FloatTensor([reward])
                
                # Calcola il valore target utilizzando la formula di Bellman
                target = reward
                if not done:
                    target += self.gamma * torch.max(self.model(next_state)).item()
                
                # Calcola i valori predetti dal modello
                current_q_values = self.model(state)
                
                # Clona i valori predetti per aggiornare il valore target per l'azione specifica
                target_f = current_q_values.clone().detach()
                target_f[0][action] = target
                # print(f"Target: {target}, Target F: {target_f}")
                
                # Calcola la perdita tra i valori predetti e i valori target
                loss = self.criterion(current_q_values, target_f)
                # print(loss)
                
                # Ottimizza il modello
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()  # Aggiorna total_loss con il valore scalare della loss
                num_transitions += 1  # Incrementa il contatore delle transizioni
            # average_loss = total_loss / num_transitions  # Calcola la media della loss
        average_loss = total_loss / num_transitions
        print(f"Average loss per episodio del minibatch: {average_loss}")
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            