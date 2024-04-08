from flax.training.train_state import TrainState
import jax.numpy as jnp
import jax




def train_step(state: TrainState):
    def loss_fn(params):
        res = state.apply_fn({"params": params}, input)
        loss = jnp.mean(res**2)
        return loss

    loss, grad = jax.value_and_grad(loss_fn)(state.params)
