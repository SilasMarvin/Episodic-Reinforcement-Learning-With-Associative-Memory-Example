import jax.numpy as jnp
import pandas as pd
import haiku as hk
import jax
import sys


"""
In practice, this may be an autoencoder, so it is designed in such a way that we are
- returning a function that can be executed with jax.apply just like an autoencoder would be
- by desining it in this way, this piece could seemlesly be replaced with an autoencoder and the
- rest of the program would not have to be changed
This reduces the continous state space into a discrete space so that memorys may overlap
"""
def create_phi(env):
    def bin(val, mx, mi):
        nval = (val - mi) / (mx - mi)
        return (jnp.round(nval, 3) * 1000).astype(jnp.int32)
    def Phi(x):
        return [
            bin(x[0], env.observation_space.high[0], env.observation_space.low[0]),  
            bin(x[1], 100, -100), #The observation space high and low are inf and -inf, but both are rarely reach (from my understanding) so 100 and -100 works fine
            bin(x[2], env.observation_space.high[2], env.observation_space.low[2]),
            bin(x[3], 100, -100), #The observation space high and low are inf and -inf, but both are rarely reach (from my understanding) so 100 and -100 works fine
        ]
    return jax.jit(Phi)



"""
A node in the memory graph
It stores the state, action, reward, next state and done value just like a normal rl memory would
- in addition, we store t (the index of the episode when this was created) and Rt (the actual value of being in this state) 
"""
class GraphNode:
    def __init__(self, embedding, state, t, action, reward, Rt, next_state, done, next_states):
        self.embedding = embedding 
        self.state = state
        self.t = t
        self.action = action
        self.reward = reward 
        self.done = done
        self.Rt = Rt
        self.next_state = next_state #The next state originally chosen by the actor, not embedded
        self.next_states = next_states 



"""
Manages the memory for the rl agent
"""
class Memory:
    def __init__(self, env, input_size):
        self.rng = jax.random.PRNGKey(42)
        phi = create_phi(env)
        self.phi = hk.transform(phi) #Our phi doesn't have any params, but if it were an autoencoder, this is how it would be used
        self.phi_params = self.phi.init(self.rng, jnp.zeros([input_size,]))
        self.e_state_mem, self.e_action_mem, self.e_reward_mem, self.e_next_state_mem, self.e_done_mem = [], [], [], [], []
        self.memory = {} 
        self.max_memory = 1000000
        self.gamma = 0.99


    #Takes a state and embeds it 
    def embed(self, state):
        return self.phi.apply(self.phi_params, self.rng, state)

    
    #Add the state, action, reward, next_state and done to the current episode memory
    def add(self, state, action, reward, next_state, done):
        self.e_state_mem.append(state), self.e_action_mem.append(action), self.e_reward_mem.append(reward), self.e_next_state_mem.append(next_state), self.e_done_mem.append(done)


    #In reference to the paper: Episodic Reinforcment learning with Associative Memory
    #This is equivalent to lines 17-21 of Algorithm 2
    def episode_end(self):
        Rt = 0
        previous_node = None
        for t in reversed(range(len(self.e_state_mem))):
            embedding = self.embed(self.e_state_mem[t])
            Rt = self.e_reward_mem[t] + self.gamma * Rt
            next_state_h = self.embed(self.e_next_state_mem[t])
            key = hash(tuple([int(x) for x in embedding])) 
            if not key in self.memory:
                self.memory[key] = GraphNode(embedding, self.e_state_mem[t], t, self.e_action_mem[t], self.e_reward_mem[t], Rt, self.e_next_state_mem[t], self.e_done_mem[t], {})
            else:
                if Rt > self.memory[key].Rt:
                    self.memory[key].Rt = Rt
            if not previous_node == None:
                self.memory[key].next_states[self.e_action_mem[t]] = previous_node
            previous_node = self.memory[key]
        if len(self.memory) > self.max_memory:
            pass #Should delete some memory, the method for deleting is not covered explicitly in the paper and is therefore left open-ended here
        self.e_state_mem, self.e_action_mem, self.e_reward_mem, self.e_next_state_mem, self.e_done_mem = [], [], [], [], []


    #In reference to the paper: Episodic Reinforcment learning with Associative Memory
    #This is equivalent to line: 23 of Algorithm 2
    def update_graph(self):
        mem_arr = list(self.memory.values())
        mem_arr.sort(reverse=True, key= lambda x: x.t)
        for mem in mem_arr:
            found = False
            max_reward = -9999999
            for x in mem.next_states.values():
                if x.Rt > max_reward and not x == mem:
                    max_reward = x.Rt
                    found = True
            if found:
                mem.Rt = mem.reward + self.gamma * max_reward


    #Samples a batch randomly from the memory
    def sample_batch(self, batch_size):
        self.rng, rng_key = jax.random.split(self.rng)
        indices = jax.random.randint(rng_key, (batch_size,), minval=0, maxval=len(self.memory))
        state_mem, action_mem, reward_mem, next_state_mem, done_mem, graph_value_mem = [], [], [], [], [], []
        mem_arr = list(self.memory.values())
        for i in indices:
            state_mem.append(mem_arr[i].state)
            action_mem.append(mem_arr[i].action)
            reward_mem.append(mem_arr[i].reward)
            next_state_mem.append(mem_arr[i].next_state)
            done_mem.append(mem_arr[i].done)
            if mem_arr[i].action in mem_arr[i].next_states:
                graph_value_mem.append(mem_arr[i].next_states[mem_arr[i].action].Rt)
            else:
                graph_value_mem.append(0)
        return (jnp.array(state_mem), jnp.array(action_mem), jnp.array(reward_mem), jnp.array(next_state_mem), jnp.array(done_mem), jnp.array(graph_value_mem))
