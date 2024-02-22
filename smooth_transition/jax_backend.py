import jax.numpy as jnp
from jax import grad, jit, vmap, random, nn, lax
import matplotlib.pyplot as plt


def get_parameters(npc, nact,key, scale=0.25, numsigma=1):
    x = jnp.linspace(-1,1,int(npc))
    xx,yy = jnp.meshgrid(x,x)
    pc_cent = jnp.concatenate([xx.reshape(-1)[:,None],yy.reshape(-1)[:,None]],axis=1)
    pc_sigma = jnp.ones([npc**2,numsigma])*scale
    pc_constant = jnp.ones(npc**2)*1.0
    actor_key, critic_key = random.split(key, num=2)
    return [jnp.array(pc_cent), jnp.array(pc_sigma), jnp.array(pc_constant), 
            1e-5 * random.normal(actor_key, (npc**2,nact)), 1e-5 * random.normal(critic_key, (npc**2,1))]


@jit
def predict_placecell(params, x):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    exponent = jnp.sum((x - pc_centers)**2 / (2 * pc_sigmas ** 2),axis=1)
    pcact = pc_constant * jnp.exp(-exponent)
    return pcact

@jit
def predict_value(params, pcact):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    value = jnp.matmul(pcact, critic_weights)
    return value


@jit
def predict_action(params, pcact, beta=2):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    actout = jnp.matmul(pcact, actor_weights)
    aprob = nn.softmax(beta * actout)
    return aprob

def get_discounted_rewards(rewards, gamma=0.95, norm=False):
    discounted_rewards = []
    cumulative = 0
    for reward in rewards[::-1]:
        cumulative = reward + gamma * cumulative  # discounted reward with gamma
        discounted_rewards.append(cumulative)
    discounted_rewards.reverse()
    if norm:
        discounted_rewards = (discounted_rewards - jnp.mean(discounted_rewards)) / (jnp.std(discounted_rewards) + 1e-9)
    return discounted_rewards


@jit
def pg_loss(params, coords, actions, discount_rewards):
    aprobs = []
    for coord in coords:
        pcact = predict_placecell(params, coord)
        aprob = predict_action(params, pcact)
        aprobs.append(aprob)
    aprobs = jnp.array(aprobs)
    neg_log_likelihood = jnp.sum(jnp.log(aprobs) * actions,axis=1)[:,None]  # log likelihood
    weighted_rewards = lax.stop_gradient(jnp.array(discount_rewards)[:,None])
    tot_loss = jnp.sum(jnp.array(neg_log_likelihood * weighted_rewards))  # log policy * discounted reward
    return tot_loss

@jit
def update_params(params, coords, actions, discount_rewards, actor_eta):
    grads = grad(pg_loss)(params, coords,actions, discount_rewards)
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    dpcc, dpcs, dpca, dact, dcri = grads

    # + for gradient ascent
    #actor_eta, critic_eta = etas
    #newpc_centers = pc_centers + pc_eta * dpcc
    #newpc_sigma = pc_sigmas + sigma_eta * dpcs
    #newpc_const = pc_constant + constant_eta * dpca
    newactor_weights = actor_weights + actor_eta * dact
    # critic weights are not updated since there is no critic for Policy Gradient method
    return [pc_centers, pc_sigmas, pc_constant, newactor_weights, critic_weights], grads


@jit
def a2c_loss(params, coords, actions, discount_rewards):
    aprobs = []
    values = []
    for coord in coords:
        pcact = predict_placecell(params, coord)
        aprob = predict_action(params, pcact)
        value = predict_value(params, pcact)
        aprobs.append(aprob)
        values.append(value)
    aprobs = jnp.array(aprobs)
    values = jnp.array(values)

    log_likelihood = jnp.sum(jnp.log(aprobs) * actions,axis=1)[:,None]  # log likelihood
    advantage = jnp.array(discount_rewards)[:,None] - values

    actor_loss = jnp.sum(log_likelihood * lax.stop_gradient(advantage))  # log policy * discounted reward
    critic_loss = -jnp.sum(advantage ** 2) # grad decent
    tot_loss = actor_loss + 0.1 * critic_loss
    return tot_loss

@jit
def update_a2c_params(params, coords, actions, discount_rewards, actor_eta, critic_eta):
    grads = grad(a2c_loss)(params, coords,actions, discount_rewards)
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    dpcc, dpcs, dpca, dact, dcri = grads

    # + for gradient ascent
    #newpc_centers = pc_centers + pc_eta * dpcc
    #newpc_sigma = pc_sigmas + sigma_eta * dpcs
    #newpc_const = pc_constant + constant_eta * dpca
    newactor_weights = actor_weights + actor_eta * dact
    newcritic_weights = critic_weights + critic_eta * dcri  # gradient descent for critic, make sure there is a negative sign in the loss
    return [pc_centers, pc_sigmas,pc_constant, newactor_weights,newcritic_weights], grads

