##-------------------------------LIBRARIES----------------------------------##
from random import random
##-----------------------------Class definition-----------------------------##
"""

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
        H = 3                       # Height 
        W = 4                       # Width
        initPos = [0, 0]            # [verticalPos,horizontalPos] for initial state
        goalPos = [H-1, W-1]        # [verticalPos,horizontalPos] for goal state
        obstPos = [[0, 2],[2, 1]]   # [verticalPos,horizontalPos] for obstacle states
        
        # Positon attributes -------------------------------------------------
        self.hPos = hPos
        self.vPos = vPos
        
        # List of possible actions -------------------------------------------
        self.actions = ["Stay","North","South","East","West"]
        
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
        
        # Obstacle uncertainty------------------------------------------------
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
         
    def __str__(self):
        return  "state-(" + str(self.vPos) + "," + str(self.hPos) + ")"
    
    def SampleAction(self):
         
        """
        This method returns a random action according to the transitions 
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
        
            
    def SampleChild(s,action):       
        
        # STAY action definition ---------------------------------------------
        if action=="Stay" :
            #In any case the successor will be the same state.
            successor = s
                         
        # NORTH action definition --------------------------------------------
        elif action=="North" :
            if s.obstacle or s.goal or s.top:
                successor = s
                            
            elif not s.top and s.left :
                r = random()           # create a random number in the range [0,1]
                if r<=0.9:
                    vPosPrime = s.vPos - 1
                    hPosPrime = s.hPos
                    
                else :
                    vPosPrime = s.vPos - 1
                    hPosPrime = s.hPos + 1
                
                successor = State(vPosPrime,hPosPrime)
                
            elif not s.top and s.right :
                r = random()           # create a random number in the range [0,1]
                if r<=0.9:
                    vPosPrime = s.vPos - 1
                    hPosPrime = s.hPos
                    
                else :
                    vPosPrime = s.vPos - 1
                    hPosPrime = s.hPos - 1
                
                successor = State(vPosPrime,hPosPrime)
                
            else : # ordinary case
                r = random()           # create a random number in the range [0,1]
                if r<=0.8:
                    vPosPrime = s.vPos - 1
                    hPosPrime = s.hPos
                elif r>0.8 and r<=0.9:
                    vPosPrime = s.vPos - 1
                    hPosPrime = s.hPos - 1
                else:
                    vPosPrime = s.vPos - 1
                    hPosPrime = s.hPos + 1
                successor = State(vPosPrime,hPosPrime)
                
        # SOUTH action definition --------------------------------------------
        elif action=="South" :
            if s.obstacle or s.goal or s.bottom:
                successor = s
                            
            elif not s.bottom and s.left :
                r = random()           # create a random number in the range [0,1]
                if r<=0.9:
                    vPosPrime = s.vPos + 1
                    hPosPrime = s.hPos
                    
                else :
                    vPosPrime = s.vPos + 1
                    hPosPrime = s.hPos + 1
                
                successor = State(vPosPrime,hPosPrime)
                
            elif not s.bottom and s.right :
                r = random()           # create a random number in the range [0,1]
                if r<=0.9:
                    vPosPrime = s.vPos + 1
                    hPosPrime = s.hPos
                    
                else :
                    vPosPrime = s.vPos + 1
                    hPosPrime = s.hPos - 1
                
                successor = State(vPosPrime,hPosPrime)
                
            else : # ordinary case
                r = random()           # create a random number in the range [0,1]
                if r<=0.8:
                    vPosPrime = s.vPos + 1
                    hPosPrime = s.hPos
                elif r>0.8 and r<=0.9:
                    vPosPrime = s.vPos + 1
                    hPosPrime = s.hPos - 1
                else:
                    vPosPrime = s.vPos + 1
                    hPosPrime = s.hPos + 1
                successor = State(vPosPrime,hPosPrime)
                
        # EAST action definition ---------------------------------------------
        elif action=="East" :
            if s.obstacle or s.goal or s.right:
                successor = s
                            
            elif not s.right and s.top :
                r = random()           # create a random number in the range [0,1]
                if r<=0.9:
                    vPosPrime = s.vPos 
                    hPosPrime = s.hPos + 1
                    
                else :
                    vPosPrime = s.vPos + 1
                    hPosPrime = s.hPos + 1
                
                successor = State(vPosPrime,hPosPrime)
                
            elif not s.right and s.bottom :
                r = random()           # create a random number in the range [0,1]
                if r<=0.9:
                    vPosPrime = s.vPos 
                    hPosPrime = s.hPos + 1
                    
                else :
                    vPosPrime = s.vPos - 1
                    hPosPrime = s.hPos + 1
                
                successor = State(vPosPrime,hPosPrime)
                
            else : # ordinary case
                r = random()           # create a random number in the range [0,1]
                if r<=0.8:
                    vPosPrime = s.vPos 
                    hPosPrime = s.hPos + 1
                elif r>0.8 and r<=0.9:
                    vPosPrime = s.vPos + 1
                    hPosPrime = s.hPos + 1
                else:
                    vPosPrime = s.vPos - 1
                    hPosPrime = s.hPos + 1
                successor = State(vPosPrime,hPosPrime)
                
        # WEST action definition ---------------------------------------------
        elif action=="West" :
            if s.obstacle or s.goal or s.left:
                successor = s
                            
            elif not s.left and s.top :
                r = random()           # create a random number in the range [0,1]
                if r<=0.9:
                    vPosPrime = s.vPos 
                    hPosPrime = s.hPos - 1
                    
                else :
                    vPosPrime = s.vPos + 1
                    hPosPrime = s.hPos - 1
                
                successor = State(vPosPrime,hPosPrime)
                
            elif not s.left and s.bottom :
                r = random()           # create a random number in the range [0,1]
                if r<=0.9:
                    vPosPrime = s.vPos 
                    hPosPrime = s.hPos - 1
                    
                else :
                    vPosPrime = s.vPos - 1
                    hPosPrime = s.hPos - 1
                
                successor = State(vPosPrime,hPosPrime)
                
            else : # ordinary case
                r = random()           # create a random number in the range [0,1]
                if r<=0.8:
                    vPosPrime = s.vPos 
                    hPosPrime = s.hPos - 1
                elif r>0.8 and r<=0.9:
                    vPosPrime = s.vPos + 1
                    hPosPrime = s.hPos - 1
                else:
                    vPosPrime = s.vPos - 1
                    hPosPrime = s.hPos - 1
                successor = State(vPosPrime,hPosPrime)
               
        cost = Cost(successor)       
        return [successor, cost]
        
        
    
    
    
    
    
    
                