# RBE595-Optimal-Control-Project-S20
This project aims to find the optimal strategy for playing a variation of the popular casino card game blackjack by formulating the game as a Markov decision process and solving it using some methods of Monte Carlo prediction and control.

## Blackjack Variation Used
In this project, I will be using the following variation of blackjack. The objective of the game is to obtain the maximum possible sum of the numerical values of your cards but not exceeding 21. All face cards count as 10, and an ace can count
either as 1 or as 11. The player competes against the dealer. The game begins with two cards dealt to both dealer and player. Only one of the dealer’s cards is face up. If the player already has sum 21 (an ace and a 10-card), he wins (this is called a natural or blackjack and is shown in Figure \ref{bjeg}) unless the dealer also has a natural, in which case the game is a draw. If the player does not have a natural, then he can request additional cards, one by one (hits), until he either stops (sticks) or exceeds 21 (goes bust). If he goes bust, he loses; if he sticks, it becomes the dealer’s turn. The dealer hits or sticks according to a fixed strategy without choice: he sticks on any sum of 17 or greater, and hits otherwise. If the dealer goes bust, then the player wins; otherwise, whoever has the final sum closer to 21 wins. This is the same as the variation used in the Sutton and Barto Reinforcement Learning text book.

## Dependencies
1) Python 3
2) numpy library
3) tqdm library
4) matplotlib library
5) seaborn library

## Code
The code is divided into 3 Python scripts.
1) main.py: This is the main file and you may use this script to run the Monte Carlo methods, and plot and save the results.
2) monte_carlo.py: This file contains a class that implements on and off policy Monte Carlo prediction and Monte Carlo control with exploring starts.
3) blackjack.py: This file contains a class that simulates a game of the blackjack variation described.

## How to Run
First, navigate to the code/ folder.
Then in the main.py script, choose which algorithms you want to run and get the results. You may comment the methods you do not wish to run.
Finally, run the main.py script. In Ubuntu, if you have Python 3 installed and is available in bash, this is simply:
```
python 3 main.py
```
## Results
### On-Policy Monte Carlo Prediction
<img src="https://github.com/ajaydxb97/RBE595-Optimal-Control-Project-S20/blob/master/figures/Pred1.png" align="middle" width=50% height=50%>
