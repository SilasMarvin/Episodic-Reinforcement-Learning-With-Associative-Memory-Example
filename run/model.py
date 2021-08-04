import jax.numpy as jnp
import pandas as pd
import haiku as hk
import optax
import jax
import sys


"""
A standard RL agent with act and update methods
It utilizes a online and target network as did the reference paper
"""
class Model:
    def __init__(self, input_size, output_size):
        def network(x):
            mlp = hk.Sequential([
              hk.Linear(300), jax.nn.relu,
              hk.Linear(100), jax.nn.relu,
              hk.Linear(output_size),
            ])
            return mlp(x)

        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = 0.01
        self.epsilon = 1
        self.epsilon_decay = 0.9994
        self.min_epsilon = 0.1
        self.gamma = 0.99
        self.update_step = 0 
        self.copy_weights_every = 20

        self.ml = hk.transform(network)
        self.rng = jax.random.PRNGKey(42)
        self.online_params = self.ml.init(self.rng, jnp.ones([input_size,]))
        self.target_params = self.online_params
        self.opt = optax.adam(self.learning_rate)
        self.opt_state = self.opt.init(self.online_params)

        #Defined here without self because it is a pure function
        def predict(rng, params, x):
            return self.ml.apply(params, rng, x)
        self._predict = jax.jit(predict)

        #Defined here without self becuase it is a pure function
        def select_action(rng, params, x, epsilon):
            return self._predict(rng, params, x)
        self._select_action = jax.jit(select_action)

        #Computes the loss in refernce to paper as defined in Equation 4
        def loss(params, rng, updated_q_values, state_mem, action_masks, graph_value_mem):
            q_values = self.ml.apply(params, rng, state_mem) #Can't call predict, grads will not be adjusted properly
            q_values = jnp.amax(jnp.multiply(q_values, action_masks), axis=1)
            return (jnp.sum(jnp.square(jnp.subtract(updated_q_values, q_values) + 0.1 * jnp.square(jnp.subtract(graph_value_mem, q_values))))) / jnp.shape(updated_q_values)[0]
        self._loss = jax.jit(loss)


    #Gets an action from the agent
    def act(self, x):
        _, self.rng = jax.random.split(self.rng)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
        if jax.random.uniform(self.rng, (1,), minval=0.0, maxval=1.0) > self.epsilon:
            return jnp.argmax(self._select_action(self.rng, self.online_params, x, self.epsilon))
        else:
            return jax.random.randint(self.rng, (1,), minval=0, maxval=self.output_size)[0]


    #Updates the agent's online network and potentially copies the online networks weights to the target network
    def update(self, batch):
        state_mem, action_mem, reward_mem, next_state_mem, done_mem, graph_value_mem = batch
        future_rewards = self._predict(self.rng, self.target_params, next_state_mem)
        updated_q_values = reward_mem + self.gamma * jnp.amax(future_rewards, axis=1)
        updated_q_values = updated_q_values * (1 - done_mem)
        action_masks = jax.nn.one_hot(action_mem, self.output_size)
        grads = jax.grad(self._loss)(self.online_params, self.rng, updated_q_values, state_mem, action_masks, graph_value_mem)
        updates, self.opt_state = self.opt.update(grads, self.opt_state)
        self.online_params = optax.apply_updates(self.online_params, updates)
        if self.update_step % self.copy_weights_every == 0:
            self.target_params = self.online_params
        self.update_step += 1
