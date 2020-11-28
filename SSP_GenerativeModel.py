##-------------------------------LIBRARIES----------------------------------##
from random import random
import math

##-----------------------------Class definition-----------------------------##
"""
The State class is basically the definition of the problem. Then, this problem 
is made of a variable number of states organized in a grid configuration. The 
size of the grid and the number of obstacles are parameters that can be easyly 
modified in the construcor of the State object.

"""

def Cost(state):
    
    nominalCost = -0.05
    obstacleCost = -5
    goalCost = 0
    
    if state.goal: return goalCost
    elif state.obstacle: return obstacleCost
    else: return nominalCost
    
class State:
    
    def __init__(self, vPos, hPos):
        # Essential information of the map -----------------------------------
        """
        H = 3                       # Height 
        W = 4                       # Width
        initPos = [0, 0]            # [verticalPos,horizontalPos] for initial state
        goalPos = [H-1, W-1]        # [verticalPos,horizontalPos] for goal state
        obstPos = [[0, 2],[2, 1]]   # [verticalPos,horizontalPos] for obstacle states
        """
        H = 4                       # Height 
        W = 10                       # Width
        initPos = [0, 0]            # [verticalPos,horizontalPos] for initial state
        goalPos = [H-1, W-1]        # [verticalPos,horizontalPos] for goal state
        obstPos = [[0, 5],[1, 9],[2,2],[3,6]]   # [verticalPos,horizontalPos] for obstacle states
        
        # Positon attributes -------------------------------------------------
        self.hPos = hPos
        self.vPos = vPos
           
        # Special States -----------------------------------------------------
        if [vPos,hPos] == initPos : self.initial = True
        else :                      self.initial = False
        
        if  [vPos,hPos] == goalPos :  self.goal = True
        else :                        self.goal = False
        
        if  [vPos,hPos] in obstPos : self.obstacle = True
        else :                       self.obstacle = False
     
        # Border detection ---------------------------------------------------
        if vPos == 0: self.top = True
        else:         self.top = False
            
        if vPos == H-1: self.bottom = True
        else:           self.bottom = False
            
        if hPos == 0: self.left = True
        else:         self.left = False
            
        if hPos == W-1: self.right = True
        else:           self.right = False
        
        # List of possible actions -------------------------------------------
        self.actions = ["Stay","North","South","East","West"]
        
        # List of Applicable actions -----------------------------------------
        self.relevActions = self.actions[1:]  # Use slice[1:] to copy the
                                              # all content (except Stay) in an
                                              # independent list
        #remove lazy actions
        if self.top    : self.relevActions.remove("North")
        if self.left   : self.relevActions.remove("West")
        if self.right  : self.relevActions.remove("East")
        if self.bottom : self.relevActions.remove("South")
        
        # Obstacle uncertainty------------------------------------------------
        """
        Aditional hardcoded attribute to test a particular Algorithm that tries
        to customize UCT's exploration coefficient...only works for 3x4 grid 
        and a particular obstacle position...
        """
        """
        if vPos ==0:
            if hPos == 0: self.uncertainty = 0
            if hPos == 1: self.uncertainty = 0.14
            if hPos == 2: self.uncertainty = 1
            if hPos == 3: self.uncertainty = 0.14
        elif vPos ==1:
            if hPos == 0: self.uncertainty = 0.66
            if hPos == 1: self.uncertainty = 0.92
            if hPos == 2: self.uncertainty = 0.92
            if hPos == 3: self.uncertainty = 0.66
        elif vPos ==2:
            if hPos == 0: self.uncertainty = 0.14
            if hPos == 1: self.uncertainty = 1
            if hPos == 2: self.uncertainty = 0.14
            if hPos == 3: self.uncertainty = 0
        else: 
            print("error")
        """
            
    def __str__(self):
        return  "state-(" + str(self.vPos) + "," + str(self.hPos) + ")"
    
    
    def SampleAction(self):
         
        """
        This method returns a random action according to the actions 
        atribute. this method is used in the rollouts of the UCT algorithm
        """
        r = random()                # sample a random number in the range [0,1]
        nA = len(self.actions)      # compute the number of actions
        deltaP = 1/nA               # compute an increment of probability
        accrual = 0                 # accrual probability
        for action in self.actions:
            
            accrual += deltaP       # addup the increment to accrual
            if r <= accrual: return action
            else: continue
        
            
    def SampleChild(self,action):
        
        """
        This is the transition model of the problem. This fuction returns a 
        succesor and the cost associated to the transition (s,a,s')
        """
        
        # STAY action definition ---------------------------------------------
        if action=="Stay" :
            #In any case the successor will be the same state.
            successor = self
                         
        # NORTH action definition --------------------------------------------
        elif action=="North" :
            if self.obstacle or self.goal or self.top:
                successor = self
                            
            elif not self.top and self.left :
                r = random()           # create a random number in the range [0,1]
                if r<=0.9:
                    vPosPrime = self.vPos - 1
                    hPosPrime = self.hPos
                    
                else :
                    vPosPrime = self.vPos - 1
                    hPosPrime = self.hPos + 1
                
                successor = State(vPosPrime,hPosPrime)
                
            elif not self.top and self.right :
                r = random()           # create a random number in the range [0,1]
                if r<=0.9:
                    vPosPrime = self.vPos - 1
                    hPosPrime = self.hPos
                    
                else :
                    vPosPrime = self.vPos - 1
                    hPosPrime = self.hPos - 1
                
                successor = State(vPosPrime,hPosPrime)
                
            else : # ordinary case
                r = random()           # create a random number in the range [0,1]
                if r<=0.8:
                    vPosPrime = self.vPos - 1
                    hPosPrime = self.hPos
                elif r>0.8 and r<=0.9:
                    vPosPrime = self.vPos - 1
                    hPosPrime = self.hPos - 1
                else:
                    vPosPrime = self.vPos - 1
                    hPosPrime = self.hPos + 1
                successor = State(vPosPrime,hPosPrime)
                
        # SOUTH action definition --------------------------------------------
        elif action=="South" :
            if self.obstacle or self.goal or self.bottom:
                successor = self
                            
            elif not self.bottom and self.left :
                r = random()           # create a random number in the range [0,1]
                if r<=0.9:
                    vPosPrime = self.vPos + 1
                    hPosPrime = self.hPos
                    
                else :
                    vPosPrime = self.vPos + 1
                    hPosPrime = self.hPos + 1
                
                successor = State(vPosPrime,hPosPrime)
                
            elif not self.bottom and self.right :
                r = random()           # create a random number in the range [0,1]
                if r<=0.9:
                    vPosPrime = self.vPos + 1
                    hPosPrime = self.hPos
                    
                else :
                    vPosPrime = self.vPos + 1
                    hPosPrime = self.hPos - 1
                
                successor = State(vPosPrime,hPosPrime)
                
            else : # ordinary case
                r = random()           # create a random number in the range [0,1]
                if r<=0.8:
                    vPosPrime = self.vPos + 1
                    hPosPrime = self.hPos
                elif r>0.8 and r<=0.9:
                    vPosPrime = self.vPos + 1
                    hPosPrime = self.hPos - 1
                else:
                    vPosPrime = self.vPos + 1
                    hPosPrime = self.hPos + 1
                successor = State(vPosPrime,hPosPrime)
                
        # EAST action definition ---------------------------------------------
        elif action=="East" :
            if self.obstacle or self.goal or self.right:
                successor = self
                            
            elif not self.right and self.top :
                r = random()           # create a random number in the range [0,1]
                if r<=0.9:
                    vPosPrime = self.vPos 
                    hPosPrime = self.hPos + 1
                    
                else :
                    vPosPrime = self.vPos + 1
                    hPosPrime = self.hPos + 1
                
                successor = State(vPosPrime,hPosPrime)
                
            elif not self.right and self.bottom :
                r = random()           # create a random number in the range [0,1]
                if r<=0.9:
                    vPosPrime = self.vPos 
                    hPosPrime = self.hPos + 1
                    
                else :
                    vPosPrime = self.vPos - 1
                    hPosPrime = self.hPos + 1
                
                successor = State(vPosPrime,hPosPrime)
                
            else : # ordinary case
                r = random()           # create a random number in the range [0,1]
                if r<=0.8:
                    vPosPrime = self.vPos 
                    hPosPrime = self.hPos + 1
                elif r>0.8 and r<=0.9:
                    vPosPrime = self.vPos + 1
                    hPosPrime = self.hPos + 1
                else:
                    vPosPrime = self.vPos - 1
                    hPosPrime = self.hPos + 1
                successor = State(vPosPrime,hPosPrime)
                
        # WEST action definition ---------------------------------------------
        elif action=="West" :
            if self.obstacle or self.goal or self.left:
                successor = self
                            
            elif not self.left and self.top :
                r = random()           # create a random number in the range [0,1]
                if r<=0.9:
                    vPosPrime = self.vPos 
                    hPosPrime = self.hPos - 1
                    
                else :
                    vPosPrime = self.vPos + 1
                    hPosPrime = self.hPos - 1
                
                successor = State(vPosPrime,hPosPrime)
                
            elif not self.left and self.bottom :
                r = random()           # create a random number in the range [0,1]
                if r<=0.9:
                    vPosPrime = self.vPos 
                    hPosPrime = self.hPos - 1
                    
                else :
                    vPosPrime = self.vPos - 1
                    hPosPrime = self.hPos - 1
                
                successor = State(vPosPrime,hPosPrime)
                
            else : # ordinary case
                r = random()           # create a random number in the range [0,1]
                if r<=0.8:
                    vPosPrime = self.vPos 
                    hPosPrime = self.hPos - 1
                elif r>0.8 and r<=0.9:
                    vPosPrime = self.vPos + 1
                    hPosPrime = self.hPos - 1
                else:
                    vPosPrime = self.vPos - 1
                    hPosPrime = self.hPos - 1
                successor = State(vPosPrime,hPosPrime)
               
        cost = Cost(successor)       
        return [successor, cost]
   
    
    
    def Entropy(self,a):        
        """
        This method returns the entropy of a particular pair state action
        defined as:
            e(s,a) = -SUM (s') {P(s'|s,a)*log2(P(s'|s,a))
    
        Note that: if a solver uses this function, it will be using some
        information about the trasition model that should be hidden from a 
        generative point of view.

        """
        # Compute 2 possibilities for the entropy according to the transition 
        # model
        e1 = -(0.8 * math.log2(0.8) + 0.2 * math.log2(0.1))
        e2 = -(0.9 * math.log2(0.9) + 0.1 * math.log2(0.1))
        
        # Return e2 if the state is in a border and the action goes parallel
        # to that border. otherwise return e1 unless the action isn't relevant
        if a not in self.relevActions : return 0
        elif self.bottom and a in ["East","West"]   : return e2
        elif self.top    and a in ["East","West"]   : return e2
        elif self.left   and a in ["North","South"] : return e2
        elif self.right  and a in ["North","South"] : return e2
        else : return e1
        
        
        
    def MaxEntropy(self):
        """
        This method returns the entropy of a particular state defined as:
            e(s) = max (a in A) {-SUM (s') {P(s'|s,a)*log2(P(s'|s,a))}
    
        Note that: if a solver uses this function, it will be using some
        information about the trasition model that should be hidden from a 
        generative point of view.

        """
       
        aux = []                     # create an empty list to store all e(s,a)
        for a in self.relevActions:  # sweep only relevant actions
        
            # append entropy
            aux.append(self.Entropy(a)) 
        
        # return the max val
        return max(aux)
    
    
    
    def MeanEntropy(self):
        """
        This method returns the entropy of a particular state defined as:
            e(s) = 1/|A| SUM(a in A) {-SUM (s') {P(s'|s,a)*log2(P(s'|s,a))}
    
        Note that: if a solver uses this function, it will be using some
        information about the trasition model that should be hidden from a 
        generative point of view.

        """    
        
        aux = []                     # create an empty list to store all e(s,a)
        for a in self.relevActions:  # sweep only relevant actions
            
        # append entropy
            aux.append(self.Entropy(a))     

        # return the average value
        return (1/len(aux)) * sum(aux)
            
            
            
            
    
    
    
    
    
                