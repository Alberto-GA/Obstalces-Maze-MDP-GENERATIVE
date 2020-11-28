import matplotlib.pyplot as plt
import statistics
from SSP_GenerativeModel import State
from UCT import UCT_like
from UCT_Custom import UCT_CustomCoefficient
from UCT_EBC import UCT_adativeCoefficient
from MCTS_LoopsBlocking import MCTS_Tplus
from maxUCT import maxUCT_like
from maxUCT_EBC import maxUCT_adaptive

##----------------------------------------------------------------------------
def CheckResult(G):
    
    rv = False                # Init the return value as False
    for s in G.keys():        # look for s(1,1) in the dictionary
        """
        if [s.vPos, s.hPos] == [1, 1]: 
            V = G[s]["V"]     
            if G[s]["East"]["Q-value"]== V : # Check if East is the best action
                rv = True
                return rv
            else:
                return rv
           
        else: continue
        """
        if [s.vPos, s.hPos] == [1, 4]: 
            V = G[s]["V"]     
            if G[s]["South"]["Q-value"]== V : # Check if south is the best action
                rv = True
                return rv
            else:
                return rv
        
##----------------------------------MAIN------------------------------------##

s0 = State(0,0)                   # Instanciate the initial state
maxTrials = 500                   # Define the number of trials
Coefficient = [0.1, 1, 5, 10]     # Define the UCB coefficients to test
N_attempts = 100                  # Define the number of times each algorithm is attempted
FinalValues = []                  # Variable to store ALL the results            
Fails = {}

# TEST UCT--------------------------------------------------------------------
Fails["UCT"] = {}

for c in Coefficient:
    
    aux = []
    attempt = 0
    failiure = 0
    while attempt < N_attempts :
        
        attempt += 1
        try:
            [Graph,Vs0] = UCT_like(s0, maxTrials,c)
            if CheckResult(Graph) : 
                aux.append(Vs0[-1])
            else: 
                failiure += 1
        except RecursionError:
            failiure += 1
            
    FinalValues.append(aux)
    Fails["UCT"][c] = failiure
    

# TEST MAXUCT-----------------------------------------------------------------       

Fails["MaxUCT"] = {}

for c in Coefficient:
    
    aux = []
    attempt = 0
    failiure = 0
    while attempt < N_attempts :
        
        attempt += 1
        try:
            [Graph,Vs0] = maxUCT_like(s0, maxTrials,c)
            if CheckResult(Graph) : 
                aux.append(Vs0[-1])
            else: 
                failiure += 1
        except RecursionError:
            failiure += 1
            
    FinalValues.append(aux)
    Fails["MaxUCT"][c] = failiure
    
    
# TEST EBC-------------------------------------------------------------------           
Fails["EBC"] = {}
option = 0
while option <= 2: 
    aux = []
    attempt = 0
    failiure = 0
    while attempt < N_attempts :
        
        attempt += 1
        try:
            [Graph,Vs0] = UCT_adativeCoefficient(s0, maxTrials,option)
            if CheckResult(Graph) : 
                aux.append(Vs0[-1])
            else: 
                failiure += 1
        except RecursionError:
            failiure += 1
            
    FinalValues.append(aux)
    
    if   option == 0 : option_str = "Max"
    elif option == 1 : option_str = "Mean"
    elif option == 2 : option_str = "Pair"
    else: print("Something strange happened")
    
    Fails["EBC"][option_str] = failiure
    option += 1      

# TEST maxEBC-----------------------------------------------------------------           
Fails["MaxEBC"] = {}
option = 0
while option <= 5: 
    aux = []
    attempt = 0
    failiure = 0
    while attempt < N_attempts :
        
        attempt += 1
        try:
            [Graph,Vs0] = maxUCT_adaptive(s0, maxTrials,option)
            if CheckResult(Graph) : 
                aux.append(Vs0[-1])
            else: 
                failiure += 1
        except RecursionError:
            failiure += 1
            
    FinalValues.append(aux)
    
    if   option == 0 : option_str = "Max"
    elif option == 1 : option_str = "Mean"
    elif option == 2 : option_str = "Pair"
    elif option == 3 : option_str = "Max_Estim"
    elif option == 4 : option_str = "Mean_Estim"
    elif option == 5 : option_str = "Pair_Estim"
    else: print("Something strange happened")
    
    Fails["MaxEBC"][option_str] = failiure
    option += 1 

# --------------------------------------------------------------------------
# Postprocessing
stats = [[],[]]
for lst in FinalValues:
    stats[0].append(abs(statistics.mean(lst)))
    stats[1].append(statistics.pstdev(lst))

success = []
success.append( (N_attempts - Fails["UCT"][0.1]) / N_attempts * 100 )
success.append( (N_attempts - Fails["UCT"][1])   / N_attempts * 100 )
success.append( (N_attempts - Fails["UCT"][5])   / N_attempts * 100 )
success.append( (N_attempts - Fails["UCT"][10])  / N_attempts * 100 )
success.append( (N_attempts - Fails["MaxUCT"][0.1]) / N_attempts * 100 )
success.append( (N_attempts - Fails["MaxUCT"][1])   / N_attempts * 100 )
success.append( (N_attempts - Fails["MaxUCT"][5])   / N_attempts *100 )
success.append( (N_attempts - Fails["MaxUCT"][10])  / N_attempts *100 )
success.append( (N_attempts - Fails["EBC"]["Max"])  / N_attempts * 100 )
success.append( (N_attempts - Fails["EBC"]["Mean"]) / N_attempts * 100 )
success.append( (N_attempts - Fails["EBC"]["Pair"]) / N_attempts * 100 )
success.append( (N_attempts - Fails["MaxEBC"]["Max"])  / N_attempts * 100 )
success.append( (N_attempts - Fails["MaxEBC"]["Mean"]) / N_attempts * 100 )
success.append( (N_attempts - Fails["MaxEBC"]["Pair"]) / N_attempts * 100 )
success.append( (N_attempts - Fails["MaxEBC"]["Max_Estim"])  / N_attempts * 100 )
success.append( (N_attempts - Fails["MaxEBC"]["Mean_Estim"]) / N_attempts * 100 )
success.append( (N_attempts - Fails["MaxEBC"]["Pair_Estim"]) / N_attempts * 100 )

#  Show the results.

# Create figure
fig = plt.figure()
# Create subplot =~ create axis
ax = fig.add_subplot(111)
# Create lables
ax.set(title="Mean and standard deviation of each algorithm", xlabel="|V(s0)|", ylabel="algorithm")
# Plot data
labels = ["UCT-0.1", "UCT-1", "UCT-5", "UCT-10", "MaxUCT-0.1", "MaxUCT-1",
          "MaxUCT-5", "MaxUCT-10", "EBC_max", "EBC_mean", "EBC_pair", 
          "maxEBC_max", "maxEBC_mean","maxEBC_pair",
          "maxEBC_max_est","maxEBC_mean_est", "maxEBC_pair_est" ]
plt.barh(labels, stats[0], color=[0.69,0.55,0.78], xerr= stats[1])

# Add a legend
plt.legend()
# Show the plot
plt.show()    

# Create figure
fig = plt.figure()
# Create subplot =~ create axis
ax = fig.add_subplot(111)
# Create lables
ax.set(title="Success rate", xlabel="% Success", ylabel="Algorithm")
# Plot data
plt.barh(labels,success , color=[0.8,1,0.8])

# Add a legend
plt.legend()
# Show the plot
plt.show()    
