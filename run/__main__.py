import gym
import jax.numpy as jnp
import sys

import model
import memory


"""
The main loop for the program 
"""
if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    mem = memory.Memory(env, input_size=4)
    ml = model.Model(input_size=4, output_size=2)

    for i in range(500): 
        state = jnp.array(env.reset())
        done = False
        step = 0
        while not done:
            action = int(ml.act(state))
            next_state, reward, done, _ = env.step(action)
            next_state = jnp.array(next_state)
            mem.add(state, action, reward, next_state, done)
            state = next_state
            step += 1
        print(f"MADE TO STEP: {step}  EPSILON: {ml.epsilon}")

        #In reference to the paper, perform lines 17-21 of Algorithm 2 
        mem.episode_end()

        #In reference to the paper, perform line 23 of Algorithm 2 
        #Please note: the paper implys that graph is updated once ever x episodes, 
        #- However we do it every episode because of the small size of our memory
        mem.update_graph()

        #In reference to the paper, perform lines 11-15 of Algorithm 2
        ml.update(mem.sample_batch(64))
