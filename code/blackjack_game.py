import numpy as np
from tqdm import tqdm

# a class that simulates a game of blackjack. Used in monte_carlo.py to generate games/episodes
class Blackjack:

    def __init__(self,dealer_policy):
        self.POLICY_DEALER = dealer_policy
        self.ACTION_HIT = 0
        self.ACTION_STICK = 1
        self.state = [] #Current state of game: a 3-tuple of usable ace (bool),player sum and dealer card
        self.player_trajectory = [] #sequence of states and actions
    # get a new card
    def get_card(self):
        card = np.random.randint(1, 14)
        card = min(card, 10)
        return card

    # get the value of a card (ace is 11).
    def card_value(self, card_num):
        if card_num == 1:
            return 11
        return card_num

    # play a game of blackjack
    # policy_player: specify policy for player
    # initial_state: [whether player has a usable Ace, sum of player's cards, one card of dealer] (optional)
    # initial_action: the initial action (optional)
    def play(self, policy_player, initial_state=None, initial_action=None):
        # player status

        # sum of player
        player_sum = 0

        # trajectory of player
        self.player_trajectory = []

        # whether player uses Ace as 11
        usable_ace_player = False

        # dealer status
        dealer_card1 = 0
        dealer_card2 = 0
        usable_ace_dealer = False

        if initial_state is None:
            # generate a random initial state

            while player_sum < 12:
                # if sum of player is less than 12, always hit
                card = self.get_card()
                player_sum += self.card_value(card)

                # If the player's sum is larger than 21, he may hold one or two aces.
                if player_sum > 21:
                    assert player_sum == 22
                    # last card must be ace
                    player_sum -= 10
                else:
                    usable_ace_player |= (1 == card)

            # initialize cards of dealer, suppose dealer will show the first card he gets
            dealer_card1 = self.get_card()
            dealer_card2 = self.get_card()

        else:
            # use specified initial state
            usable_ace_player, player_sum, dealer_card1 = initial_state
            dealer_card2 = self.get_card()

        # initial state of the game
        self.state = [usable_ace_player, player_sum, dealer_card1]

        # initialize dealer's sum
        dealer_sum = self.card_value(dealer_card1) + self.card_value(dealer_card2)
        usable_ace_dealer = 1 in (dealer_card1, dealer_card2)
        # if the dealer's sum is larger than 21, he must hold two aces.
        if dealer_sum > 21:
            assert dealer_sum == 22
            # use one Ace as 1 rather than 11
            dealer_sum -= 10
        assert dealer_sum <= 21
        assert player_sum <= 21

        # game starts!

        # player's turn
        while True:
            if initial_action is not None:
                action = initial_action
                initial_action = None
            else:
                # get action based on current sum
                action = policy_player(usable_ace_player, player_sum, dealer_card1)

            # track player's trajectory for importance sampling
            self.player_trajectory.append([(usable_ace_player, player_sum, dealer_card1), action])

            if action == self.ACTION_STICK:
                break
            # if hit, get new card
            card = self.get_card()
            # Keep track of the ace count. the usable_ace_player flag is insufficient alone as it cannot
            # distinguish between having one ace or two.
            ace_count = int(usable_ace_player)
            if card == 1:
                ace_count += 1
            player_sum += self.card_value(card)
            # If the player has a usable ace, use it as 1 to avoid busting and continue.
            while player_sum > 21 and ace_count:
                player_sum -= 10
                ace_count -= 1
            # player busts
            if player_sum > 21:
                return self.state, -1, self.player_trajectory
            assert player_sum <= 21
            usable_ace_player = (ace_count == 1)

        # dealer's turn
        while True:
            # get action based on current sum
            action = self.POLICY_DEALER[dealer_sum]
            if action == self.ACTION_STICK:
                break
            # if hit, get a new card
            new_card = self.get_card()
            ace_count = int(usable_ace_dealer)
            if new_card == 1:
                ace_count += 1
            dealer_sum += self.card_value(new_card)
            # If the dealer has a usable ace, use it as 1 to avoid busting and continue.
            while dealer_sum > 21 and ace_count:
                dealer_sum -= 10
                ace_count -= 1
            # dealer busts
            if dealer_sum > 21:
                return self.state, 1, self.player_trajectory
            usable_ace_dealer = (ace_count == 1)

        # compare the sum between player and dealer
        assert player_sum <= 21 and dealer_sum <= 21
        if player_sum > dealer_sum:
            return self.state, 1, self.player_trajectory
        elif player_sum == dealer_sum:
            return self.state, 0, self.player_trajectory
        else:
            return self.state, -1, self.player_trajectory