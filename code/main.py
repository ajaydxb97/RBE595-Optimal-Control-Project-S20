import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from monte_carlo import MonteCarlo

# actions: hit or stick
ACTION_HIT = 0 #request additional card
ACTION_STICK = 1  #stop requesting additional cards
CONTROLS = [ACTION_HIT, ACTION_STICK] #control inputs

# test player policy for player used in MC on policy prediction
# stick only if player sum is >= 20, else hit
POLICY_PLAYER = np.zeros(22, dtype=np.int)
for i in range(12, 20):
    POLICY_PLAYER[i] = ACTION_HIT
POLICY_PLAYER[20] = ACTION_STICK
POLICY_PLAYER[21] = ACTION_STICK

# another test player policy for player used in MC on policy prediction
# stick only if player sum is >= 18, else hit
POLICY_PLAYER2 = np.zeros(22, dtype=np.int)
for i in range(12, 18):
    POLICY_PLAYER2[i] = ACTION_HIT
for i in range(18, 22):
    POLICY_PLAYER2[i] = ACTION_STICK

# fixed policy for dealer
# stick only if dealer sum is >= 17, else hit
POLICY_DEALER = np.zeros(22)
for i in range(12, 17):
    POLICY_DEALER[i] = ACTION_HIT
for i in range(17, 22):
    POLICY_DEALER[i] = ACTION_STICK

#----------------------------------Result Functions---------------------------------------------------

def MC_OnPolicy_Prediction_Results():
    mc_obj = MonteCarlo(POLICY_PLAYER2,POLICY_DEALER)

    states_usable_ace_1, states_no_usable_ace_1 = mc_obj.monte_carlo_on_policy(10000)
    states_usable_ace_2, states_no_usable_ace_2 = mc_obj.monte_carlo_on_policy(1000000)

    states = [states_usable_ace_1,
              states_usable_ace_2,
              states_no_usable_ace_1,
              states_no_usable_ace_2]

    titles = ['Usable Ace, 10000 Episodes',
              'Usable Ace, 1000000 Episodes',
              'No Usable Ace, 10000 Episodes',
              'No Usable Ace, 1000000 Episodes']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    sns.set(font_scale=3)
    for state, title, axis in zip(states, titles, axes):
        fig = sns.heatmap(np.flipud(state), cmap="YlGnBu", ax=axis)
        fig.set_yticklabels(list(reversed(range(12, 22))), fontsize=35)
        fig.set_xticklabels(range(1, 11), fontsize=35)
        fig.set_ylabel('Player sum', fontsize=40)
        fig.set_xlabel('Dealer showing', fontsize=40)
        fig.set_title(title, fontsize=40)

    plt.savefig('MC_OnPolicy_Prediction.png')
    plt.close()

def MC_OnPolicy_Control_Results():
    mc_obj = MonteCarlo(POLICY_PLAYER,POLICY_DEALER)

    state_action_values = mc_obj.monte_carlo_es_control(500000)

    state_value_no_usable_ace = np.max(state_action_values[:, :, 0, :], axis=-1)
    state_value_usable_ace = np.max(state_action_values[:, :, 1, :], axis=-1)

    # get the optimal policy
    action_no_usable_ace = np.argmax(state_action_values[:, :, 0, :], axis=-1)
    action_usable_ace = np.argmax(state_action_values[:, :, 1, :], axis=-1)

    images = [action_usable_ace,
              state_value_usable_ace,
              action_no_usable_ace,
              state_value_no_usable_ace]

    titles = ['Optimal policy with usable Ace',
              'Optimal value with usable Ace',
              'Optimal policy without usable Ace',
              'Optimal value without usable Ace']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    sns.set(font_scale=3)
    for image, title, axis in zip(images, titles, axes):
        fig = sns.heatmap(np.flipud(image), cmap="YlGnBu", ax=axis)
        fig.set_yticklabels(list(reversed(range(12, 22))), fontsize=35)
        fig.set_xticklabels(range(1, 11), fontsize=35)
        fig.set_ylabel('Player sum', fontsize=40)
        fig.set_xlabel('Dealer showing', fontsize=40)
        fig.set_title(title, fontsize=40)

    plt.savefig('MC_OnPolicy_Control.png')
    plt.close()

def MC_OffPolicy_Prediction_Results():
    mc_obj = MonteCarlo(POLICY_PLAYER,POLICY_DEALER)

    true_value = -0.27726
    episodes = 10000
    runs = 100
    error_ordinary = np.zeros(episodes) #MSE of ordinary importance sampling
    error_weighted = np.zeros(episodes) #MSE of weighted importance sampling
    for i in tqdm(range(0, runs)):
        ordinary_sampling_, weighted_sampling_ = mc_obj.monte_carlo_off_policy(episodes)
        # get the squared error
        error_ordinary += np.power(ordinary_sampling_ - true_value, 2)
        error_weighted += np.power(weighted_sampling_ - true_value, 2)
    error_ordinary /= runs
    error_weighted /= runs

    plt.plot(error_weighted, label='Weighted Importance Sampling')
    plt.plot(error_ordinary, label='Ordinary Importance Sampling')
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Mean square error')
    plt.xscale('log')
    plt.legend()

    plt.savefig('MC_OffPolicy_Prediction.png')
    plt.close()


if __name__ == '__main__':
    #Running this will save the output plots for each Monte Carlo method implemented
	MC_OnPolicy_Prediction_Results()
    #MC_OnPolicy_Control_Results()
    #MC_OffPolicy_Prediction_Results()