import numpy as np
from tqdm import tqdm
from blackjack_game import Blackjack

# a class with Monte Carlo methods implemented
class MonteCarlo:

    def __init__(self, player_policy, dealer_policy):
        self.POLICY_PLAYER = player_policy
        self.POLICY_DEALER = dealer_policy
        self.ACTION_HIT = 0 # request additional card
        self.ACTION_STICK = 1 # stop requesting additional cards
        self.CONTROLS = [self.ACTION_HIT,self.ACTION_STICK]

    # function form of target policy of player
    def target_policy_player(self, usable_ace_player, player_sum, dealer_card):
        return self.POLICY_PLAYER[player_sum]

    # function form of behavior policy of player
    def behavior_policy_player(self, usable_ace_player, player_sum, dealer_card):
        if np.random.binomial(1, 0.5) == 1:
            return self.ACTION_STICK
        return self.ACTION_HIT

    # Monte Carlo Sample with On-Policy
    def monte_carlo_on_policy(self, episodes):
        game = Blackjack(self.POLICY_DEALER)
        states_usable_ace = np.zeros((10, 10))
        # initialze counts to 1 to avoid 0 being divided
        states_usable_ace_count = np.ones((10, 10))
        states_no_usable_ace = np.zeros((10, 10))
        # initialze counts to 1 to avoid 0 being divided
        states_no_usable_ace_count = np.ones((10, 10))
        for i in tqdm(range(0, episodes)):
            _, reward, player_trajectory = game.play(self.target_policy_player)
            for (usable_ace, player_sum, dealer_card), _ in player_trajectory:
                player_sum -= 12
                dealer_card -= 1
                if usable_ace:
                    states_usable_ace_count[player_sum, dealer_card] += 1
                    states_usable_ace[player_sum, dealer_card] += reward
                else:
                    states_no_usable_ace_count[player_sum, dealer_card] += 1
                    states_no_usable_ace[player_sum, dealer_card] += reward
        return states_usable_ace / states_usable_ace_count, states_no_usable_ace / states_no_usable_ace_count

    # Monte Carlo with Exploring Starts
    def monte_carlo_es_control(self, episodes):
        game = Blackjack(self.POLICY_DEALER)
        # (playerSum, dealerCard, usableAce, action)
        state_action_values = np.zeros((10, 10, 2, 2))
        # initialze counts to 1 to avoid division by 0
        state_action_pair_count = np.ones((10, 10, 2, 2))

        # behavior policy is greedy
        def behavior_policy(usable_ace, player_sum, dealer_card):
            usable_ace = int(usable_ace)
            player_sum -= 12
            dealer_card -= 1
            # get argmax of the average returns(s, a)
            values_ = state_action_values[player_sum, dealer_card, usable_ace, :] / \
                      state_action_pair_count[player_sum, dealer_card, usable_ace, :]
            return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

        # play for several episodes
        for episode in tqdm(range(episodes)):
            # for each episode, use a randomly initialized state and action
            initial_state = [bool(np.random.choice([0, 1])),
                           np.random.choice(range(12, 22)),
                           np.random.choice(range(1, 11))]
            initial_action = np.random.choice(self.CONTROLS)
            current_policy = behavior_policy if episode else self.target_policy_player
            _, reward, trajectory = game.play(current_policy, initial_state, initial_action)
            first_visit_check = set()
            for (usable_ace, player_sum, dealer_card), action in trajectory:
                usable_ace = int(usable_ace)
                player_sum -= 12
                dealer_card -= 1
                state_action = (usable_ace, player_sum, dealer_card, action)
                if state_action in first_visit_check:
                    continue
                first_visit_check.add(state_action)
                # update values of state-action pairs
                state_action_values[player_sum, dealer_card, usable_ace, action] += reward
                state_action_pair_count[player_sum, dealer_card, usable_ace, action] += 1

        return state_action_values / state_action_pair_count

    # Monte Carlo Sample with Off-Policy
    def monte_carlo_off_policy(self, episodes):
        initial_state = [True, 13, 2]

        rhos = []
        returns = []

        game = Blackjack(self.POLICY_DEALER)

        for i in range(0, episodes):
            _, reward, player_trajectory = game.play(self.behavior_policy_player, initial_state=initial_state)

            # get the importance ratio
            numerator = 1.0
            denominator = 1.0
            for (usable_ace, player_sum, dealer_card), action in player_trajectory:
                if action == self.target_policy_player(usable_ace, player_sum, dealer_card):
                    denominator *= 0.5
                else:
                    numerator = 0.0
                    break
            rho = numerator / denominator
            rhos.append(rho)
            returns.append(reward)

        rhos = np.asarray(rhos)
        returns = np.asarray(returns)
        weighted_returns = rhos * returns

        weighted_returns = np.add.accumulate(weighted_returns)
        rhos = np.add.accumulate(rhos)

        ordinary_sampling = weighted_returns / np.arange(1, episodes + 1)

        with np.errstate(divide='ignore',invalid='ignore'):
            weighted_sampling = np.where(rhos != 0, weighted_returns / rhos, 0)

        return ordinary_sampling, weighted_sampling