# create task
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# define Place Cell --> Actor-Critic agent
class PC_AC_agent:
    def __init__(self, npc=21,pcr=0.25, nact=4, alr=0.0075, clr=0.025):
        self.npc = npc  # number of place cells tiling each dimension
        self.alr = alr  # actor learning rate
        self.clr = clr  # critic learning rate
        self.pcspacing = np.linspace(-1,1,self.npc) # uniformly space place cells
        self.pcr =  pcr # define radius of place cells

        xx, yy = np.meshgrid(self.pcspacing, self.pcspacing)
        self.pcs = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1)],axis=1)

        self.nact = nact  # number of action units
        self.wC = np.random.normal(loc=0,scale=0.001, size=[len(self.pcs), 1]) #np.zeros([len(self.pcs), 1])  # critic weight matrix
        self.wA = np.random.normal(loc=0,scale=0.001, size=[len(self.pcs), nact]) #np.zeros([len(self.pcs), nact])  # actor weight matrix
        self.gamma = 0.95  # discount factor
        self.beta = 2  # action temperature hyperparameters, higher --> more exploitation
    
    def get_pc(self, x):
        # convert x,y coordinate to place cell activity
        norm = np.sum((x - self.pcs)**2,axis=1)
        pcact = np.exp(-norm / (2 * self.pcr **2))
        return pcact
    
    def softmax(self, prob):
        return np.exp(prob) / np.sum(np.exp(prob))
    
    def get_action(self, x):
        # get place cell activity
        self.h = self.get_pc(x)

        # get critic activity
        self.V = np.matmul(self.h, self.wC)

        # get actor activity
        self.A = np.matmul(self.h, self.wA)

        # choose action using stochastic policy
        self.prob = self.softmax(self.beta* self.A)
        A = np.random.choice(np.arange(self.nact), p=self.prob)

        # convert action to onehot
        self.onehotg = np.zeros(self.nact)
        self.onehotg[A] = 1
        return self.onehotg

    def learn(self, newstate, reward):
        # get value estimate of new state using current critic weights     
        self.V1 = np.matmul(self.get_pc(newstate), self.wC)

        # compute TD error 
        self.td = int(reward) + self.gamma * self.V1 - self.V

        # update weights at each timestep when TD is computed using 2 & 3 factor Hebbian rule
        self.wC += self.clr * self.h[:,None] * self.td
        self.wA += self.alr * np.matmul(self.h[:,None], self.onehotg[:,None].T) * self.td
        
    
    def plot_maps(self, env, title=None):
        plt.figure()
        plt.title(title)
        plt.imshow(self.wC.reshape(self.npc, self.npc), origin='lower')
        plt.colorbar()
        dir = np.matmul(self.wA, env.onehot2dirmat)
        xx, yy = np.meshgrid(np.arange(self.npc), np.arange(self.npc))
        plt.quiver(xx.reshape(-1),yy.reshape(-1), dir[:,0], dir[:,1], color='w', scale_units='xy',scale=None)
        plt.show()



# environment
class TwoDimNav:
    def __init__(self,obstacles=False, maxspeed=0.1, envsize=1, goalsize=0.1, seed=2023, tmax=100, goalcoord=[0,0.8], startcoord='corners') -> None:
        self.tmax = tmax  # maximum steps per trial
        self.minsize = -envsize  # arena size
        self.maxsize = envsize
        self.state = np.zeros(2)
        self.done = False
        self.goalsize = goalsize

        self.statesize = 2 # start + goal information
        self.goal = np.array(goalcoord)

        if startcoord =='corners':  # agent starts from one of 4 corners
            self.starts = np.array([[-0.8,-0.8], [-0.8,0.8], [0.8,0.8], [0.8,-0.8]],dtype=np.float32)
        elif startcoord == 'center':
            self.starts = np.array([0.0,0.0])  # agent starts from the center
        else:
            self.starts = np.array(startcoord)
            print(startcoord)

        self.actionsize = 4
        self.maxspeed = maxspeed  # max agent speed per step

        self.obstacles = obstacles
        self.e = 0 if seed is None else seed

        # convert agent's onehot vector action to direction in the arena
        self.onehot2dirmat = np.array([
            [0,1],  # up
            [1,0],  # right
            [0,-1],  # down
            [-1,0]  # left
        ])
    
    def action2velocity(self, g):
        # convert onehot action vector from actor to velocity
        return np.matmul(g, self.onehot2dirmat)

    
    def reset(self):
        self.e +=1

        if len(self.starts) > 2:
            np.random.seed(self.e)
            startidx = np.random.choice(np.arange(len(self.starts)),1)
            self.state = self.starts[startidx].copy()[0]
        else:
            self.state = self.starts.copy()
        
        print(self.state)

        self.error = self.goal - self.state
        self.eucdist = np.linalg.norm(self.error,ord=2)
        self.done = False
        self.t = 0

        self.track = []
        self.track.append(self.goal.copy())
        self.track.append(self.state.copy())

        self.actions = np.zeros(self.statesize)

        print(f"State: {self.state}, Goal: {self.goal}")
        return self.state, self.goal, self.error, self.done

    
    def step(self, g):
        self.t +=1
        velocity = self.action2velocity(g)  # get velocity from agent's onehot action
        self.actions += 0.25 * velocity  # smoothen actions so that agent explores the entire arena. From Foster et al. 2000
        newstate = self.state.copy() + self.actions * self.maxspeed  # update state with action velocity

        self.track.append(self.state.copy())

        # check if new state crosses boundary
        if (newstate > self.maxsize).any() or (newstate < self.minsize).any():
            newstate = self.state.copy()
            self.actions = np.zeros(self.statesize)

        # check if new state crosses obstacles if initalized
        if self.obstacles:
            if  -0.6 < newstate[0] < -0.4 and  0.3 < newstate[1] < 1:  # top left obs
                newstate = self.state.copy()
                self.actions = np.zeros(self.statesize)
            
            if  -0.6 < newstate[0] < -0.4 and  -1 < newstate[1] < -0.3:  # bottom left obs
                newstate = self.state.copy()
                self.actions = np.zeros(self.statesize)
            
            if  0.4 < newstate[0] < 0.6 and  0.3 < newstate[1] < 1:  # top right obs
                newstate = self.state.copy()
                self.actions = np.zeros(self.statesize)
            
            if  0.4 < newstate[0] < 0.6 and  -1 < newstate[1] < -0.3:  # top right obs
                newstate = self.state.copy()
                self.actions = np.zeros(self.statesize)
        
        # if new state does not violate boundary or obstacles, update new state
        self.state = newstate.copy()
        self.error = self.goal - self.state
        self.eucdist = np.linalg.norm(self.error,ord=2)

        # check if agent is within radius of goal
        if self.eucdist < self.goalsize:
            self.done = True

        return self.state, self.error, self.done

    def random_action(self):
        action = np.random.uniform(low=-1, high=1,size=self.statesize)
        return action 

    def plot_trajectory(self, title=None):
        plt.figure()
        plt.title(f'2D {title}')
        plt.axis([self.minsize, self.maxsize, self.minsize, self.maxsize])
        plt.grid()

        if self.obstacles:
            plt.gca().add_patch(Rectangle((-0.6,0.3), 0.2, 0.7, facecolor='grey'))  # top left
            plt.gca().add_patch(Rectangle((0.4,0.3), 0.2, 0.7, facecolor='grey'))  # top right
            plt.gca().add_patch(Rectangle((-0.6,-0.3), 0.2, -0.7, facecolor='grey'))  # bottom left
            plt.gca().add_patch(Rectangle((0.4,-0.3), 0.2, -0.7, facecolor='grey'))  # bottom right

        #plt.scatter(np.array(self.track)[0,0],np.array(self.track)[0,1], color='r', zorder=2, )    
        circle = plt.Circle(xy=self.goal, radius=self.goalsize, color='r')
        plt.gca().add_patch(circle)
        plt.scatter(np.array(self.track)[1,0],np.array(self.track)[1,1], color='g', zorder=2)    
        plt.plot(np.array(self.track)[1:,0],np.array(self.track)[1:,1], marker='o',color='b', zorder=1)

        plt.gca().set_aspect('equal')
        
        