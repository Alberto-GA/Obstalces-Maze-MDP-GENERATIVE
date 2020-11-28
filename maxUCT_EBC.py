"""
         from UCT_EBC  ->  NEW BACKUP FUNCTION  ->  MAX UCT_EBC


"""
#-------------------------------LIBRAIRES------------------------------------#
import math
import operator
#-------------------------------FUNCTIONS------------------------------------#
"""
"""
def Rollout(s):
    
    depth = 5       # Define the depth parameter, how deep do you want to go?
    nRollout = 0    # initialise the rollout counter
    payoff = 0      # initialise the cummulative cost/reward
    while nRollout < depth:
        
        # Stop the rollout if a dead-end is reached.
        # NOTE: "the first state will never be a dead-end so payoff not 0"
        if s.obstacle : return payoff
        
        # The rollouts progress with random actions -> sample an action
        a = s.SampleAction()

        # Sample a state according to P(s'|s,a)
        [successor, cost] = s.SampleChild(a)
        
        # Compute the inmediate cost/reward and update the payoff
        payoff += cost
       
        # update the current state with the sampled successor
        s = successor
        
        # increase the rollout counter
        nRollout += 1
        
    return payoff

#----------------------------------------------------------------------------#    
"""
Three different methods to sample an action according to UCB-EBC. All of them 
will only take into account relevant actions... if some experiments must be 
undertaken considering all actions, please replace s.relevActions by s.actions

"""
def ActionSelection_Max(s,G):
    c = [0,2]           # Exploration coefficient bounds 
    UCB = {}            # Dictionary to save the result of UCB for each action
    
    # Compute normalised entropy with MaxEntropy
    en = (c[1]-c[0])*s.MaxEntropy() + c[0]
    # Compute the adaptive explotration coefficient by rescaling with the 
    # higher cost/reward
    c_ebc = 5*en 
    # CONSIDER ONLY RELEVANT ACTIONS 
    for a in s.relevActions:
    
        # Modified UCB formula                              
        UCB[a] = G[s][a]["Q-value"] + c_ebc * math.sqrt(math.log(G[s]["N"])/G[s][a]["Na"])

    # choose the relevant action that maximize the UCB formula    
    a_UCB = max(UCB.items(), key=operator.itemgetter(1))[0]
    return a_UCB

"""

"""
def ActionSelection_Mean(s,G):
    c = [0,2]           # Exploration coefficient bounds 
    UCB = {}         # Dictionary to save the result of UCB for each action
    
            
    # Compute normalised entropy with MeanEntropy
    en = (c[1]-c[0])*s.MeanEntropy() + c[0]
    # Compute the adaptive explotration coefficient by rescaling with the 
    # higher cost/reward
    c_ebc = 5*en 
        
    # CONSIDER ONLY RELEVANT ACTIONS 
    for a in s.relevActions:

        # Modified UCB formula                              
        UCB[a] = G[s][a]["Q-value"] + c_ebc * math.sqrt(math.log(G[s]["N"])/G[s][a]["Na"])

    # choose the relevant action that maximize the UCB formula    
    a_UCB = max(UCB.items(), key=operator.itemgetter(1))[0]
    return a_UCB

"""

"""
def ActionSelection_Pair(s,G):
    c = [0,2]           # Exploration coefficient bounds 
    UCB = {}         # Dictionary to save the result of UCB for each action
    
    # CONSIDER ONLY RELEVANT ACTIONS 
    for a in s.relevActions:
        
        # Compute normalised entropy based s.Entropy(a)
        en = (c[1]-c[0])*s.Entropy(a) + c[0]
        # Compute the adaptive explotration coefficient by rescaling with the 
        # higher cost/reward
        c_ebc = 5*en 
        # Modified UCB formula                              
        UCB[a] = G[s][a]["Q-value"] + c_ebc * math.sqrt(math.log(G[s]["N"])/G[s][a]["Na"])

    # choose the relevant action that maximize the UCB formula    
    a_UCB = max(UCB.items(), key=operator.itemgetter(1))[0]
    return a_UCB

"""
The previous functions took advantage of some domain-dependent information. 
They open a small window to see what is inside the transition model. All in all
they are like cheating and breaking the spirit of a generative model. The 
following functions will try to accomplish the same tasks but using estimated 
probabilities to compute the entropy of a state-action pair. 
Warning!! at the begining there are not enough sampled data and the transitions
may seem deterministic. This leads to a low entropy and consequently to a low 
exploration... I have a bad feeling about it...but let's see how it works.
"""

def ActionSelection_Pair_Estimated(s,G):
    c = [0,2]        # Exploration coefficient bounds 
    UCB = {}         # Dictionary to save the result of UCB for each action
    
    # CONSIDER ONLY RELEVANT ACTIONS 
    for a in s.relevActions:
        
        # Estimate the entropy, if the action hasn't a list of successors big
        # enough, let's take entropy =1 to push for exploration.  
        if len(G[s][a]["Successors"])>=2: 
            e = 0
            for child in G[s][a]["Successors"].keys():
                prob = G[s][a]["Successors"][child]/G[s][a]["Na"]
                e += - prob * math.log2(prob)
        else : 
            e = 1
        # Compute normalised entropy based on the estimated entropy
        en = (c[1]-c[0])* e + c[0]
        # Compute the adaptive explotration coefficient by rescaling with the 
        # higher cost/reward
        c_ebc = 5*en 
        # Modified UCB formula                              
        UCB[a] = G[s][a]["Q-value"] + c_ebc * math.sqrt(math.log(G[s]["N"])/G[s][a]["Na"])

    # choose the relevant action that maximize the UCB formula    
    a_UCB = max(UCB.items(), key=operator.itemgetter(1))[0]
    return a_UCB

"""

"""

def ActionSelection_Mean_Estimated(s,G):
    c = [0,2]        # Exploration coefficient bounds 
    UCB = {}         # Dictionary to save the result of UCB for each action
    
    # CONSIDER ONLY RELEVANT ACTIONS 
    # Estimate the mean entropy
    entropies = []
    for a in s.relevActions:
        e = 0
        for child in G[s][a]["Successors"].keys():
            prob = G[s][a]["Successors"][child]/G[s][a]["Na"]
            e += - prob * math.log2(prob)
        entropies.append(e)
    
    e = (1/len(entropies)) * sum(entropies)
    # Compute normalised entropy based on the estimated entropy
    en = (c[1]-c[0])* e + c[0]
    # Compute the adaptive explotration coefficient by rescaling with the 
    # higher cost/reward
    c_ebc = 5*en 
    
    for a in s.relevActions:         
        # Modified UCB formula                              
        UCB[a] = G[s][a]["Q-value"] + c_ebc * math.sqrt(math.log(G[s]["N"])/G[s][a]["Na"])

    # choose the relevant action that maximize the UCB formula    
    a_UCB = max(UCB.items(), key=operator.itemgetter(1))[0]
    return a_UCB

"""

"""

def ActionSelection_Max_Estimated(s,G):
    c = [0,2]        # Exploration coefficient bounds 
    UCB = {}         # Dictionary to save the result of UCB for each action
    
    # CONSIDER ONLY RELEVANT ACTIONS 
    # Estimate the max entropy
    entropies = []
    for a in s.relevActions:
        e = 0
        for child in G[s][a]["Successors"].keys():
            prob = G[s][a]["Successors"][child]/G[s][a]["Na"]
            e += - prob * math.log2(prob)
        entropies.append(e)
    
    e = max(entropies)
    # Compute normalised entropy based on the estimated entropy
    en = (c[1]-c[0])* e + c[0]
    # Compute the adaptive explotration coefficient by rescaling with the 
    # higher cost/reward
    c_ebc = 5*en 
    
    for a in s.relevActions:         
        # Modified UCB formula                              
        UCB[a] = G[s][a]["Q-value"] + c_ebc * math.sqrt(math.log(G[s]["N"])/G[s][a]["Na"])

    # choose the relevant action that maximize the UCB formula    
    a_UCB = max(UCB.items(), key=operator.itemgetter(1))[0]
    return a_UCB

    
#----------------------------------------------------------------------------#
    
#----------------------------------------------------------------------------#
"""
"""
def checkState(s):
    
    global G              # Get access to the graph
    # state by state check if the position of the analysed state matches with 
    # the positon of already visited states.
    for state in G.keys():  
        if [s.vPos,s.hPos] == [state.vPos, state.hPos] :
            # Overwrite s because it is a new instance 
            # of an already visited state
            s = state 
    return s

#----------------------------------------------------------------------------#        

def initNode(s):
    global G
    
    # Create a new node in the graph if this is a new state
    G[s] = {}        # intialise node's dictionary
    G[s]["N"] = 0    # Count the first visit to the node (as the number of initialised actions) 
    G[s]["V"] = 0    # Initialise the Value function of the decission Node
    
    # Initialise the Q-values based on rollouts
    # NOTE that (all the possible/only relevant) actions are tested.
    # NOTE that the childs are not created in the graph.
    aux = []          # empty list to ease the maximization
    for a in s.relevActions:
        
        # Count the initialisation of this action as a visit to Node s
        G[s]["N"] += 1 
        
        # Sample a successor according to the generative model
        [successor, cost]= s.SampleChild(a)
        
        # Create a dictionary to store a lot of information
        G[s][a]={}
        G[s][a]["Cost"] = cost                # Init expected cost of C(s,a)
        G[s][a]["Successors"] = {}            # Keep track of the children of s
        G[s][a]["Q-value"] = cost + Rollout(successor)
        aux.append(G[s][a]["Q-value"])  
        G[s][a]["Na"] = 1               # Register the visit for this pair s-a
            
    # Compute the Qvalue of the decision node (V(s)).
    G[s]["V"] = max(aux)  
    aux = []          # clear the auxiliary list
    
    #Return and finish the trial.
    rv = G[s]["V"]       # the return value is the max Q(s,a)  
    return rv

#-----------------------------------------------------------------------------    
              
def Trial(s,option):
    
    global G           # Make sure that I have access to the graph
    K = -5             # Internal parameter -> asociated cost to dead-ends
    
    
    # 0) CHECK IF THE CURRENT STATE IS NEW -----------------------------------
    # If this state have been visited before, overwrite it with the first
    # instance of that state. Otherwise continue an initialise the node.
    s = checkState(s)
    
    # 1) CHECK IF THE STATE IS TERMINAL---------------------------------------        
    if s.goal :
        if s not in G :                 #Include goal node in the Graph
            G[s] = {}
            G[s]["V"] = 0
            G[s]["N"] = 1
            return
        else :
            G[s]["N"] += 1
            return
        
    elif s.obstacle:          
        
        if s not in G :                #Include obstacle node in the Graph
            G[s] = {}
            G[s]["V"] = K
            G[s]["N"] = 1
            return
        else :
            G[s]["N"] += 1
            return         
        
            
    # 2) CHECK IF THE STATE IS ALREADY IN THE GRAPH
    if s not in G: return initNode(s)
    
    # 3) EXPAND THE NODE IF IT'S ALREADY IN THE GRAPH ------------------------
    if   option == 0 : a_UCB = ActionSelection_Max(s,G)    
    elif option == 1 : a_UCB = ActionSelection_Mean(s,G)
    elif option == 2 : a_UCB = ActionSelection_Pair(s,G)
    elif option == 3 : a_UCB = ActionSelection_Max_Estimated(s,G)
    elif option == 4 : a_UCB = ActionSelection_Mean_Estimated(s,G)
    elif option == 5 : a_UCB = ActionSelection_Pair_Estimated(s,G)
    
    
    # 4) SAMPLE A CHILD  PLAYING THIS ACTION ---------------------------------
    [successor,cost] = s.SampleChild(a_UCB)
    
    successor = checkState(successor)
    # 6) UPDATE THE COUNTERS -------------------------------------------------
    G[s]["N"] += 1
    G[s][a_UCB]["Na"] += 1
    
    if successor in G[s][a_UCB]["Successors"]:   
        G[s][a_UCB]["Successors"][successor] += 1    
    else :                           
        G[s][a_UCB]["Successors"][successor] = 1
        
    # 5) CONTINUE THE TRIAL---------------------------------------------------     
    Trial(successor,option)  
    

    # 7) BACK-UP FUNCTIONS --------------------------------------------------- 
    G[s][a_UCB]["Cost"] += (cost - G[s][a_UCB]["Cost"] ) /  G[s][a_UCB]["Na"]
    
    
    aux = 0
    if G[s][a_UCB]["Successors"]:

        for child in G[s][a_UCB]["Successors"].keys():
        
            aux += G[s][a_UCB]["Successors"][child] * G[child]["V"]
        
    
    G[s][a_UCB]["Q-value"] = G[s][a_UCB]["Cost"] + (aux) / G[s][a_UCB]["Na"] 
    
    
    # 8) UPDATE THE VALUE FUNCTION OF THE DECISION NODE------------------------
    # V(s) <- max Q(s,a) | a in A
    aux = []              
    for a in G[s].keys(): 
        if a=="N" or a=="V": continue
        else : aux.append(G[s][a]["Q-value"])
    G[s]["V"] = max(aux)
    aux = []
               
    return
    
#----------------------------------------------------------------------------#    
"""

"""
def maxUCT_adaptive(s0, maxTrials, option):
    
    nTrial = 0                         # initialize the trial counter
    global G                           # make a global variable so that all 
                                       # the functions can modify it
    G = {}                             # initialize a graph
    Vs0 = []
    
    # safety check
    """
    if option not in [0,1,2,3,4,5]:
        print("invalid option argument")
        return
    """
    
    while nTrial < maxTrials :         # perform trials while possible
        
        nTrial += 1
        Trial(s0, option)
        Vs0.append(G[s0]["V"])  
        
    return G,Vs0     