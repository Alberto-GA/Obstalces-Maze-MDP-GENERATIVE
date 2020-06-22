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
The action selection method is the heart of UCT. It is the way the algorithm 
deals with the exploration-exploitation dilemma. Namely, UCT takes the ideas 
of bandit problems and applies the UCB formula to solve the conflict. In this
formula there is a term that votes for exploitation of the best current policy.
By contrast, the other term is devoted to exploring less visited nodes.

It receives the current Graph as input because this function doesn't modify 
the graph.

"""
def ActionSelection(s,G):
    c = 10           # Exploration coefficient 
    UCB = {}         # Dictionary to save the result of UCB for each action
    
    for a in s.actions:          # UCB formula
        UCB[a] = G[s][a]["Q-value"] + c * math.sqrt(math.log(G[s]["N"])/G[s][a]["Na"])

    # choose the action that maximize the UCB formula    
    a_UCB = max(UCB.items(), key=operator.itemgetter(1))[0]
    return a_UCB

"""
There is a problem with the action selection when "lazy" actions are chosen 
from the UCB formula because the successor is the same state and this issue 
comes into conflict with the recursivity. For this reason I've suggested to 
take into account only relevant actions using the function below. 

WARNING: This choice doesn't seem to be a great idea because the nodes are 
completely initialised. This means that all the actions are tried and they
are provided with a first stimate of the Q-value(s,a) thanks to a rollout.
And where is the problem? Well, this values are initialised but the actions 
are no longer taken into account through action selection (UCB) resulting in
a biased situation between relevant and lazy actions.

CONCLUSION: if you want to remove the "lazy" actions to kill loops of type 
s'=s, only relevant actions must be initialised!!! -> LOOK STEP 2

"""
def RelevantActionSelection(s,G):
    c = [0,2]           # Exploration coefficient bounds 
    UCB = {}         # Dictionary to save the result of UCB for each action
    
    for a in s.actions:
        # CONSIDER ONLY RELEVANT ACTIONS -> avoid infinite loops between 
        # action Selection and child sampling in the recursive call. However, 
        # this doesn't prevent from loops in the tree: e.g s0-s1-s5-s4-s0...
        if a=="Stay":                    continue               
        elif s.top    and a=="North":    continue    
        elif s.right  and a=="East":     continue    
        elif s.left   and a=="West":     continue     
        elif s.bottom and a=="South":    continue
        else :                                   # UCB formula
            en = (c[1]-c[0])*s.uncertainty +c[0]
            c_ebc = 5*en
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
    
              
def Trial(s):
    
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
        for a in s.actions:
            
            if a=="Stay":                    continue    #Remove lazy actions           
            elif s.top    and a=="North":    continue    #if you want to con-
            elif s.right  and a=="East":     continue    #sider only relevant
            elif s.left   and a=="West":     continue    #actions. Remove if-else
            elif s.bottom and a=="South":    continue    #to consider all actions
            else :
              
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
        for a in  s.transitions.keys():
            G[s]["V"]+= G[s][a]["Na"] * G[s][a]["Q-value"]/G[s]["N"]
        """    
        # OPTION2: Taking into account only the optimal Q(s,a)
        #          V(s) <- max(Q(s,a)) | a in A
        G[s]["V"] = max(aux)  
        
        #Return and finish the trial.
        rv = max(aux)        # the return value is the max Q(s,a)
        aux = []             # clear the auxiliary list
        return rv
    
    # 3) EXPAND THE NODE IF IT'S ALREADY IN THE GRAPH ------------------------
        # To expand a node, UCT applies the action selection  
        # strategy that is based on the UCB formula, this code provide two
        # different functions to return the 'best' action:
        #    -Actionselection(s,G)-> all actions, including "lazy" actions, are considered.
        #    -RelevantActionSelection(s,G)-> only relevant actions are considered.
    a_UCB = RelevantActionSelection(s,G)
    
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
            
        QvaluePrime =  cost + Trial(successor)  
        
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
    G[s][a_UCB]["Q-value"] += (QvaluePrime - G[s][a_UCB]["Q-value"]) / G[s][a_UCB]["Na"]
    
    
    # 8) UPDATE THE VALUE FUNCTION OF THE DECISSION NODE
    
    # OPTION 1: V(s) <- SUM[Na(s,a) . Q(s,a)]/N(s)
    """
    G[s]["V"] = 0
    for a in  s.transitions.keys():
        G[s]["V"] += G[s][a]["Na"]*G[s][a]["Q-value"]/G[s]["N"]
    
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
def UCT_adativeCoefficient(s0, maxTrials):
    
    nTrial = 0                         # initialize the trial counter
    global G                           # make a global variable so that all 
                                       # the functions can modify it
    G = {}                             # initialize a graph
    Vs0 = 0
    while nTrial < maxTrials :         # perform trials while possible
        
        nTrial += 1
        Trial(s0)       
    
        
    Vs0 = G[s0]["V"]    
    return G,Vs0      