##-------------------------------LIBRARIES----------------------------------##
import matplotlib.pyplot as plt
import numpy as np
from SSP_GenerativeModel import State
from UCT import UCT_like
from UCT_Custom import UCT_CustomCoefficient
from UCT_EBC import UCT_adativeCoefficient
from MCTS_LoopsBlocking import MCTS_Tplus
from maxUCT import maxUCT_like
from maxUCT_EBC import maxUCT_adaptive


##----------------------------------MAIN------------------------------------##

s0 = State(0,0)           # Instanciate the initial state
maxTrials = 1000          # Define the number of trials
Coefficient = 10          # Define the UCB coefficients to test

# It is interesting to test several times the solver algorithm because of its
# stochastic nature. For this reason: firstly an empty list is created to 
# store the final result of each attempt. And secondly a for loop is built to 
# attempt and store. 
A = []                    
for i in range(0,1):
    [Graph,Vs0]= maxUCT_adaptive(s0, maxTrials, 5)
    A.append(Vs0[-1])


##--------------------------------CREATE PLOT-------------------------------##
# This section displays a plot showing the evolution of the Value of the ini-
# tial state (only last attempt)


# Create figure
fig = plt.figure()
# Create subplot =~ create axis
ax = fig.add_subplot(111)
# Create lables
ax.set(title="Evolution of the Value function", xlabel="Trials", ylabel="V")
# Plot data
ax.plot(Vs0, label='V(s0)', color=[0.3,0.2,0.9], linewidth=1)
# Add a legend
plt.legend()
# Show the plot
plt.show()


##----------------------------SAVE RESULTS AS .TXT--------------------------##
# This section writes the list A in a .txt to perform future statistical
# studies and compare the performances of the algorithms.

"""
a = np.array(A)
mat = np.matrix(a)
with open('resultsV3.txt','wb') as f:
    for line in mat.transpose():
        np.savetxt(f, line, fmt='%.16f')
        
f.close()        
"""