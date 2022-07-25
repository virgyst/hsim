# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 16:15:38 2022

@author: Lorenzo
"""

from typing import List, Any, Optional, Callable
import logging 
from simpy import Process, Interrupt, Event
from simpy.events import PENDING, Initialize, Interruption
from core import Environment, dotdict, Interruption, method_lambda
import types
from stores import Store
from collections import OrderedDict
import warnings
import pandas as pd
import copy
import dill

def function(instance):
    def decorator(f):
        f = types.MethodType(f, instance)
        setattr(instance, '_function', f)
        return f
    return decorator

def on_entry(instance):
    def decorator(f):
        f = types.MethodType(f, instance)
        instance._entry_callbacks.append(f)
        return f
    return decorator

def on_exit(instance):
    def decorator(f):
        f = types.MethodType(f, instance)
        instance._exit_callbacks.append(f)
        return f
    return decorator

def on_interrupt(instance):
    def decorator(f):
        f = types.MethodType(f, instance)
        instance._interrupt_callbacks.append(f)
        return f
    return decorator

def trigger(instance):
    def decorator(f):
        f = types.MethodType(f, instance)
        instance._trigger = f
        return f
    return decorator

def prova(instance):
    def decorator(f):
        f = types.MethodType(f, instance)
        setattr(instance, '_generator', f)
        return f
    return decorator

def do(instance):
    def decorator(f):
        f = types.MethodType(f, instance)
        if f.__code__.co_argcount < 2:
            raise TypeError('Probably missing trigger event') 
        setattr(instance, '_do', f)
        return f
    return decorator

@staticmethod
def set_state(name,initial_state=False):
    state=State(name)
    setattr(StateMachine,name,state)
    StateMachine.add_state(state,initial_state)

def add_states(sm,states):
    sm._states = states # [copy.deepcopy(state) for state in states] 
    
class StateMachine():
    def __init__(self, env, name=None):
        self.env = env
        self.var = dotdict()
        if name==None:
            self._name = str('0x%x' %id(self))
        else:
            self._name = name
        self._current_state = None
        self._build_states()
        self.start()
        self.env.add_object(self)
    def __getattr__(self,attr):
        for state in object.__getattribute__(self,'_states'):
            if state._name == attr:
                return state
        raise AttributeError()
    def __repr__(self):
        return '<%s (%s object) at 0x%x>' % (self._name, type(self).__name__, id(self))
    def start(self):
        for state in self._states:
            if state.initial_state == True:
                state.start()
        if not any([state.initial_state for state in self._states]):
            print('Warning: no initial state set')
    def interrupt(self):
        for state in self._states:
            state.interrupt()
    def stop(self):
        return self.interrupt()
    def _build_states(self):
        self._states = copy.deepcopy(self._states)
        for state in self._states:
            state.set_parent_sm(self)
    # def copy_states(self):
    #     for element in dir(self):
    #         x = getattr(self, element)
    #         if type(x) == State:
    #             x.set_parent_sm(self)
    #             self.add_state(x)
    @property
    def name(self):
        return self._name
    @property
    def current_state(self):
        return [state for state in self._states if state.is_alive]
    @property
    def is_alive(self):
        if self.current_state == []:
            return False
        else:
            return True
    @classmethod
    def _states_dict(self,state):
        list_by_name = [s for s in self._states if s.name == state]
        if list_by_name is not []:
            return list_by_name[0]
    

class CompositeState(StateMachine):
    def __init__(self, name=None):
        if name==None:
            self._name = str('0x%x' %id(self))
        else:
            self._name = name
        self._current_state = None
        self.parent_state = None
    def start(self):
        self.env = self.parent_state.env
        self._build_states()
        super().start()

class State(Process):
    def __init__(self, name, initial_state=False):
        self._name = name
        self._time = None
        self._entry_callbacks = []
        self._exit_callbacks = []
        self._child_state_machine = None
        self.sm = None
        self._interrupt_callbacks = []
        self._function = lambda self: None
        self.initial_state = initial_state
        self.callbacks = []
        self._value = None
        self._transitions = list()
    def __getattr__(self,attr):
        try:
            sm = self.__getattribute__('sm')
            return getattr(sm,attr)
        except:
            return object.__getattribute__(self,attr)
    def __repr__(self):
        return '<%s (State) object at 0x%x>' % (self._name, id(self))
    def __call__(self):
        return self.start()
    @property
    def name(self):
        return self._name
    def set_composite_state(self, compositeState):
        compositeState.parent_state = self
        self._child_state_machine = compositeState
    def set_parent_sm(self, parent_sm):
        # if not isinstance(parent_sm, StateMachine):
        #     raise TypeError("parent_sm must be the type of StateMachine")
        if self._child_state_machine and self._child_state_machine == parent_sm:
            raise ValueError("child_sm and parent_sm must be different")
        self.sm = parent_sm
    def start(self):
        logging.debug(f"Entering {self._name}")
        self._last_state_record = [self.sm,self.sm._name,self,self._name,self.env.now,None]
        self.env.state_log.append(self._last_state_record)
        for callback in self._entry_callbacks:
            callback()
        if self._child_state_machine is not None:
            self._child_state_machine.start()
        self._do_start()
    def stop(self):
        logging.debug(f"Exiting {self._name}")
        self._last_state_record[-1] = self.env.now
        for callback in self._exit_callbacks:
            callback()
        if self._child_state_machine is not None:
            self._child_state_machine.stop()
        self._do_stop()
    def _do_start(self):
        self.callbacks = []
        self._value = PENDING
        self._target = Initialize(self.env, self)
    def _do_stop(self):
        self._value = None
    def interrupt(self):
        if self.is_alive:
            Interruption(self, None)
            for callback in self._interrupt_callbacks:
                callback()
            if self._child_state_machine is not None:
                self._child_state_machine.stop()
        else:
            print('Warning - interrupted state was not active')
    def _resume(self, event):
        self.env._active_proc = self
        if isinstance(event,Initialize):
            method_lambda(self,self._function)
            events = list()
            for transition in self._transitions:
                transition._state = self
                event = transition()
                events.append(event)
        else:
            for transition in self._transitions:
                transition.cancel()
            if event is None:
                event = self
                self._do_start()
                return
            elif isinstance(event,State):
                self.stop()
                event()
            elif isinstance(event,Interruption):
                event = None
                self._ok = True
                self._value = None
                self.callbacks = []
                self.env.schedule(self)
            
            '''
                elif event._ok:
                    try:
                        event = self._do(event)
                    except:
                        warnings.warn('Error in %s (%s)' %(self,self.sm))
                        raise StopIteration
                    if event is None:
                        event = self
                        self._do_start()
                        return
                    else:
                        if type(event) is type(self):
                            self.stop()
                            event()
                        else:
                            event()
                            raise StopIteration
                elif isinstance(event,Interruption):
                    event = None
                    self._ok = True
                    self._value = None
                    self.callbacks = []
                    self.env.schedule(self)
                    break
                else:
                    event._defused = True
                    exc = type(event._value)(*event._value.args)
                    exc.__cause__ = event._value
                    event = self._generator.throw(exc)
                    warnings.warn('%s' %self)
            except StopIteration as e:
                event = None
                self._ok = True
                self._value = e.args[0] if len(e.args) else None
                self.callbacks = []
                self.env.schedule(self)
                self.stop()
                break
            except BaseException as e:
                event = None
                self._ok = False
                tb = e.__traceback__
                e.__traceback__ = tb.tb_next
                self._value = e
                self.callbacks = []
                self.env.schedule(self)
                break
            try:
                if event.callbacks is not None:
                    event.callbacks.append(self._resume)
                    break
            except AttributeError:
                if not hasattr(event, 'callbacks'):
                    msg = 'Invalid yield value "%s"' % event
                descr = self._generator #_describe_frame(self._generator.gi_frame)
                error = RuntimeError('\n%s%s' % (descr, msg))
                error.__cause__ = None
                raise error
                '''
        self._target = event
        self.env._active_proc = None


class CHFSM(StateMachine):
    def __init__(self,env,name=None):
        super().__init__(env,name)
        self._list_messages()
        self.connections = dict()
    def __getattr__(self,attr):
        for state in object.__getattribute__(self,'_states'):
            if state._name == attr:
                return state
        if object.__getattribute__(self,'_messages').__contains__(attr):
            return object.__getattribute__(self,'_messages')[attr]
        raise AttributeError()
    def build(self):
        pass
    def _associate(self):
        for state in self._states:
            state.connections = self.connections
            state.var = self.var
            for message in self._messages:
                setattr(state,message,self._messages[message])
    def _list_messages(self):
        self._messages = OrderedDict()
        temp=list(self.__dict__.keys())
        self.build()
        for i in list(self.__dict__.keys()):
            if i not in temp:
                self._messages[i] = getattr(self,i)

class Transition():
    def __init__(self, state, target=None, trigger=None, condition=None, action=None):
        self._state = state
        self._target = target
        if trigger is not None:
            self._trigger = trigger
        if action is not None:
            self._action = action
    def __getattr__(self,attr):
        try:
            state = self.__getattribute__('_state')
            try:
                return getattr(state,attr)
            except:
                sm = state.__getattribute__('sm')
                return getattr(sm,attr)
        except:
            return object.__getattribute__(self,attr)
    def _trigger(self):
        pass
    def _condition(self):
        return True
    def _action(self):
        return None
    def _otherwise(self):
        return self()
    def cancel(self):
        self._event.callbacks = []
    def _evaluate(self,event):
        if method_lambda(self,self._condition):
            method_lambda(self,self._action)
            self._state._resume(self._target)
        else:
            self._otherwise()
    def __call__(self):
        if self._trigger is None:
            return self._evaluate(None)
            self._target._state = self._state
        self._event = method_lambda(self,self._trigger)
        self._event.callbacks.append(self._evaluate)
        return self._event
 
class Pseudostate(State):
    def __init__(self):
        pass
    def _resume(self,event):
        events = list()
        for transition in self._transitions:
            transition._state = self._state
            event = transition()
            events.append(event)

if __name__ == "__main__" and 1:
    class Boh(StateMachine):
        def build(self):
            Idle = State('Idle',True)
            @function(Idle)
            def printt(self):
                print('%s is Idle' %self.sm._name)
                return self.env.timeout(10)
            @do(Idle)
            def todo(self,Event):
                print('%s waited 10s' %self.sm._name)
            @on_exit(Idle)
            def print_ciao(self):
                print('Idle state exit')
            @on_interrupt(Idle)
            def interrupted_ok(self):
                print('%s idle state interrupted ok'  %self.sm._name)
            class Idle_SM(CompositeState):
                Sub = State('Sub',True)
                @function(Sub)
                def printt(self):
                    print('%s will print this something in 20 s'  %self.sm._name)
                    return self.env.timeout(20)
                @do(Sub)
                def todo(self,Event):
                    print('Printing this only once')
                    raise
                @on_exit(Sub)
                def print_ciao(self):
                    print('Substate exit')
            Idle.set_composite_state(Idle_SM)
            return [Idle]
    
    class Boh2(CHFSM):
        def build(self):
            Work = State('Work',True)
            @function(Work)
            def printt(self):
                print('Start working. Will finish in 10s')
                return self.env.timeout(10)
            @do(Work)
            def d(self,Event):
                print("Finished!")
                return Work
            @on_exit(Work)
            def exiting(self):
                print('Leaving working state')
            @on_entry(Work)
            def entering(self):
                print('Entering working state')
            return [Work]
        
    class Boh3(CHFSM):
        pass
    Work = State('Work',True)
    @function(Work)
    def printt(self):
        print('Start working. Will finish in 10s')
        return self.env.timeout(10)
    @do(Work)
    def d(self,Event):
        print("Finished!")
        return self.Work
    add_states(Boh3,[Work])
    
    class Boh4(CHFSM):
        pass
    Work = State('Work',True)
    Work._function = lambda self:print('Start working. Will finish in 10s')
    t = Transition(Work, None, lambda self: self.env.timeout(10))
    Work._transitions = [t]
    add_states(Boh4,[Work])
    
    class Boh5(CHFSM):
        pass
    class WorkSM(CompositeState):
        pass
    Work = State('Work',True)
    Work._function = lambda self:print('Start working. Will finish in 10s')
    t = Transition(Work, None, lambda self: self.env.timeout(10))
    Work._transitions = [t]
    
    Work0 = State('Work0',True)
    Work0._function = lambda self:print('Start working 0. Will finish in 5s')
    t = Transition(Work0, None, lambda self: self.env.timeout(5))
    Work0._transitions = [t]
    add_states(WorkSM,[Work0])
    Work.set_composite_state(WorkSM('WorkSM'))
    add_states(Boh5,[Work])
    
    env = Environment()
    foo = Boh5(env,1)
    env.run(20)
    foo.interrupt()
    env.run(200)

    




    
