"""
              Second enhancement of UCT like algorithm V2

This is the second attempt to improve the performances of UCT. The approach of 
this algorithm is based on tuning the exploration coefficient in accordance with
an entropic criteria. There are several ways to define the entropy of a state, 
and some of them are considered here. However, all of them agree that the 
entropy of a particular state is higher when the outcome of an action is more 
uncertain. All in all, in this first approach, the solver will compute the en-
tropy using information about the domain so it is not using a pure generative
model. In future approaches we will try to estimate this entropy with enough 
sampled data.

Again, the only difference with UCT relies on the way the action selection is
made.

Note: this code can be used with different grid sizes.


"""
#-------------------------------LIBRAIRES------------------------------------#
import math
import operator
#-------------------------------FUNCTIONS------------------------------------#
"""
The Rollout function is used to initialise the Q-value of a new node in the 
Graph. It basically returns an estimation of the long term cost/reward starting
from the child "s".
Note that the rollout do not need to end in the goal.
Note also that the childs are never included in the graph... 
state.SampleChild(a) instansciates locally a successor state but it is not
stored in the graph.
"""
def Rollout(s):
    
    depth = 5       # Define the depth parameter, how deep do you want to go?
    nRollout = 0    # initialise the rollout counter
    payoff = 0      # initialise the cummulative cost/reward
    while nRollout < depth:
        
        # Stop the rollout if a dead-end is reached.
        # NOTE: "the first state will never be a dead-end so payoff>0"
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
    
#----------------------------------------------------------------------------#
"""
checkState(s): is a function that allows us to check if the current "new" 
state "s" is actually new or have been visited before. How does it work? 
Simple, it takes a state as an imput, if there is an state in the graph with 
the same position s is overwritten with that state and the previous instance 
is never used again. Otherwise, the function returns the state object without
any modification to inlcude it in the graph.
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
    
              
def Trial(s,option):
    
    global G           # Make sure that I have access to the graph
    K = -5             # Internal parameter -> asociated cost to dead-ends
    
    # 1) CHECK IF THE STATE IS TERMINAL---------------------------------------
        # as a reminder: in finite horizion MDP terminal means that the final
        # decision epoch has been reached. In infinte horizon (disc. reward) 
        # MDP, the terminal states are the goals and the dead-ends.
        
    if s.goal : return 0               # No cost to reach the goal
    elif s.obstacle: return K          # Penalty for dead-ends
        
    # 2) CHECK IF THE CURRENT STATE IS NEW -----------------------------------
        # If this state have been visited before, overwrite it with the first
        # instance of that state. Otherwise continue an initialise the node.
    s = checkState(s)
    
    if s not in G:
        #print("New state detected -> Initialisation")
        # Create a new node in the graph if this is a new state
        G[s] = {}        # intialise node's dictionary
        G[s]["N"] = 0    # Count the first visit to the node (as the number of initialised actions) 
        G[s]["V"] = 0    # Initialise the Value function of the decission Node
        
        # Initialise the Q-values based on rollouts
        # NOTE that (all the possible/only relevant) actions are tested.
        # NOTE that the childs are not created in the graph.
        aux = []          # empty list to ease the maximization
        
        for a in s.relevActions: # ¡¡ Use s.relevActions to remove the init of
                                 # lazy actions. Use s.actions instead for a 
                                 #  complete initialisation !!
            
            # Count the initialisation of this action as a visit to Node s
            G[s]["N"]+=1 
            
            # Sample a successor according to the generative model
            [successor, cost]= s.SampleChild(a)
            
            # the Qvalue is the inmediate cost/reward plus the long term
            # cost/reward that is estimated through a rollout
            G[s][a]={}
            G[s][a]["Q-value"] = cost + Rollout(successor)
            aux.append(G[s][a]["Q-value"])  
            
            # Register the visit for this pair s-a
            G[s][a]["Na"]= 1
                  
                
        # Compute the Qvalue of the decision node (V(s)).Two approaches are valid.
        # OPTION1: Averaging the Qvalues ofits successor chance nodes.
        #          V(s) <- SUM[Na(s,a) . Q(s,a)]/N(s)
        """
        """    
        # OPTION2: Taking into account only the optimal Q(s,a)
        #          V(s) <- max(Q(s,a)) | a in A
        G[s]["V"] = max(aux)  
        
        #Return and finish the trial.
        rv = max(aux)        # the return value is the max Q(s,a)
        aux = []             # clear the auxiliary list
        return rv
    
    # 3) EXPAND THE NODE IF IT'S ALREADY IN THE GRAPH ------------------------
    
    if   option == 0 : a_UCB = ActionSelection_Max(s,G)    
    elif option == 1 : a_UCB = ActionSelection_Mean(s,G)
    elif option == 2 : a_UCB = ActionSelection_Pair(s,G)
    
    # 4) SAMPLE A CHILD  PLAYING THIS ACTION ---------------------------------
    [successor,cost] = s.SampleChild(a_UCB)
    
    # 5) COMPUTE AN ESTIMATE OF Q(s,a_UCB)------------------------------------
        # The importance of this first estimate is twofold. First it will be 
        # used to Compute the final estimate of the Q-value. Second, its
        # recursive architecture allows to expand the state including the 
        # child in the graph, and it also performs a subsequent backup in 
        # reverse order so the trial finishes when the backup is done in the 
        # root node.
    
    if successor == s :
        
        # Kill possible loop (s'=s) with current Qvalue estimate
        # This condition never applies if only relevant actions are considered        
        QvaluePrime = cost + G[s]["V"]
        
    else :
            
        QvaluePrime =  cost + Trial(successor,option)  
        
    # 6) UPDATE THE COUNTERS -------------------------------------------------
        # The order between this step and step 5 could be reversed. This
        # is so because the target problem allows to play actions that lead
        # the agent to the same state. Taking into account the recursivity of
        # the following step, it could generate an infinte loop of 
        # actionSelection-childSampling if G is not modified so that the UCB
        # formula is affected. 
        # The objective of this strategy is not to remove the loops but to 
        # make the loops finite. To do it, the "lazy" action mustn't be the
        # result of the action selection (UCB) forever. Updating the counters
        # in combination with a high enough exploration coefficient seems to 
        # be a promising strategy...   
    G[s]["N"] += 1
    G[s][a_UCB]["Na"] += 1   
    
    # 7) UPDATE THE Q-VALUE OF THE PAIR (s,a_UCB)-----------------------------
    
        # OPTION 1 : classical POMCP-GO approach
    G[s][a_UCB]["Q-value"] += (QvaluePrime - G[s][a_UCB]["Q-value"]) / G[s][a_UCB]["Na"]
    
        # OPTION 2 : MinPOMCP-GO approach
    
    
    
    # 8) UPDATE THE VALUE FUNCTION OF THE DECISSION NODE
    
    # OPTION 1: V(s) <- SUM[Na(s,a) . Q(s,a)]/N(s)
    """
        This is not the best approach
    """
    # OPTION 2: V(s) <- max Q(s,a) | a in A
    aux = []              
    for a in G[s].keys(): 
        if a=="N" or a=="V": continue
        else : aux.append(G[s][a]["Q-value"])
    G[s]["V"] = max(aux)
    aux = []
             
    
    return QvaluePrime 
    
#----------------------------------------------------------------------------#    
"""
This is the skeleton of the UCT: it relies on the UCT_Trial mnethod wich will
update and refine the information in G, a global variable which represents
the current partial tree. The desired architecture for this variable is:
    
    G = { s1: {a1 : {"Q-value" : current estimation for Q(s1,a1)
                        "Na"   : number of times we have played a1 in s1}
               a2 : {...}
               N  : Number of times this State has been visited
               V  : Value function in the decission Node s. This computation 
                    is not essential but could be useful if "lazy" actions are playing
               } 
         
         s2:{...}
         }
"""
def UCT_adativeCoefficient(s0, maxTrials, option):
    
    nTrial = 0                         # initialize the trial counter
    global G                           # make a global variable so that all 
                                       # the functions can modify it
    G = {}                             # initialize a graph
    Vs0 = []
    
    # safety check
    """
    print("option:" , option)
    if option == 0:
        print("Adaptive coefficient based on Max Entropy")
    elif option == 1:
        print("Adaptive coefficient based on Mean Entropy")
    elif option == 2:
        print("Adaptive coefficient based on state-action pairs Entropy") 
    else: 
        print("Option error, choose an integer in [0,2]")
        return
    """
    
    while nTrial < maxTrials :         # perform trials while possible
        
        nTrial += 1
        Trial(s0,option)       
        Vs0.append(G[s0]["V"])  
        
    return G,Vs0      