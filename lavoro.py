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
import time

class Entity:
    def __init__(self,ID=None):
        self.ID = ID
        self.rework = False
        self.serviceTime = dict()
        # self.pt['M3'] = 1
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
            return 3.5 ### qua andrà messo 10/20 volte maggiore degli altri processing time?
        else:
            return super().calculateServiceTime(entity,attribute)
    def completed(self):
        if self.var.entity.ok:
            self.controller.Messages.put(self.name)
    T2=Transition(pym.Server.Working, pym.Server.Blocking, lambda self: self.env.timeout(self.calculateServiceTime(self.var.entity)), action = lambda self: self.completed())
    T3=Transition(pym.Server.Blocking, pym.Server.Starving, lambda self: self.Next.put(self.var.entity),action=lambda self: [self.var.request.confirm(), self.sm.var.entity.done() if self.sm._name=='robot' else None])

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
    def __init__(self,env):
        self.real = True
        self.capacity = 30
        self.lab = None
        self.initialWIP = 12
        self.targetWIP = 12
        self.request = None
        self.message = env.event()
        self.WIP = 0
        self.WIPlist = list()
        super().__init__(env)
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
            #Scheduling Agent
            
            self.request = self.Store.get()
            # print(type(self.request))
        try:
            self.Next.put(self.request.value)
            self.request = None
            self.WIP += 1
            self.WIPlist.append([self.env.now,self.WIP])
        except:
            pass
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
        
class Conveyor(pym.Conveyor):
    def __init__(self,env,name=None,capacity=3):
        super().__init__(env,name,capacity,0.75)
        
def newDT():
    lab = globals()['lab']
    deepcopy(lab)

def newEntity():
    seed(time.time())
    e = Entity()
    e.serviceTime['front'] = 10.52
    e.serviceTime['drill'] = choices([30, 40, 50, 20],weights=[5,30,30,35])[0]
    e.serviceTime['robot'] = choices([0, 81, 105, 108 ,120],weights=[91,3,2,2,2])[0]
    e.serviceTime['camera'] = 3.5+expovariate(1/7.1)
    e.serviceTime['back'] = choices([3.5,10.57],weights=[0.1,0.9])[0]
    if e.serviceTime['back']>0:
        e.serviceTime['press'] = 3.5+expovariate(1/7.5)
    else:
        e.serviceTime['press'] = 3.5
    e.serviceTime['manual'] = max(np.random.normal(9.2,2),0)
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

class Lab:
    def __init__(self):
        conveyTime = 6
        self.env = pym.Environment() #crea l'ambiente
        # self.g = Generator(self.env) #genera un nuovo pezzo
        self.gate = Gate(self.env) #da togliere ma ci sarà bisogno di modifiche #gestisce l'ingresso dei pezzi
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
        
        #self.g.Next = self.gate
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
        # return pd.DataFrame(self.env.state_log)
        return self.env.state_log

    def calculate_makespan(self,state_log):
        df = pd.DataFrame(state_log, columns=["Resource","ResourceName","State","StateName","Entity","?","timeIn","timeOut"])
        df= df.loc[df.ResourceName.isin(["front","drill","robot","camera","back","press","manual"])]
        mks=df.timeOut.max()-df.timeIn.min()
        return mks


# import sys
# sys.path.insert(0,'C:/Users/Lorenzo/Dropbox (DIG)/Tesisti/Giovanni Zanardo/Python')
# from Foresighted_DT_function import BN_detection

# BN_detection(lab.env.state_log,0,lab.env.now)

class Result:
    def __init__(self,time,BN,OR,DR,production,arrivals,WIPlist,BNlist,state_log):
        self.time=time
        self.BN=BN
        self.OR=OR
        self.DR=DR
        self.production=production
        self.arrivals=arrivals
        self.WIPlist = WIPlist
        self.state_log=state_log
        self.BNlist = BNlist
    @property
    def avgWIP(self):
        integral = 0.0
        prev_time = None
        prev_value = None
        # Iterate over the data points and compute the integral
        for time, value in self.WIPlist:
            if prev_time is not None and prev_value is not None:
                dt = time - prev_time
                integral += prev_value * dt
            prev_time = time
            prev_value = value
        return integral/time
    @property
    def productivity(self):
        return (3600/pd.DataFrame(self.arrivals).diff()).describe()
    @property
    def CI(self):
        prod = self.productivity
        tinv = stats.t.ppf(1-0.05/2, prod[0][0])
        return prod[0][1] - prod[0][2]*tinv/np.sqrt(prod[0][0]), prod[0][1] + prod[0][2]*tinv/np.sqrt(prod[0][0])
