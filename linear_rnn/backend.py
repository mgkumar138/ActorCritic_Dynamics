# create task
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def get_onehot_action(prob, nact=4):
    A = np.random.choice(a=np.arange(nact), p=np.array(prob))
    onehotg = np.zeros(nact)
    onehotg[A] = 1
    return onehotg

def plot_maps(env, npc, actor_weights,critic_weights=None, title='Agent map'):
    plt.figure()
    plt.title(title)
    xx, yy = np.meshgrid(np.arange(npc), np.arange(npc))
    plt.quiver(xx.reshape(-1),yy.reshape(-1), actor_weights[0], actor_weights[1], color='k', scale_units='xy')
    plt.gca().set_aspect('equal')
    plt.show()

# environment
class TwoDimNav:
    def __init__(self,obstacles=False, maxspeed=0.1, envsize=1, goalsize=0.1, tmax=100, goalcoord=[0,0.8], startcoord='corners') -> None:
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

        self.actionsize = 4
        self.maxspeed = maxspeed  # max agent speed per step

        self.obstacles = obstacles

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
        if len(self.starts) > 1:
            startidx = np.random.choice(np.arange(len(self.starts)),1)
            self.state = self.starts[startidx].copy()[0]
        else:
            self.state = self.starts.copy()
        
        self.error = self.goal - self.state
        self.eucdist = np.linalg.norm(self.error,ord=2)
        self.done = False
        self.t = 0

        self.track = []
        self.track.append(self.goal.copy())
        self.track.append(self.state.copy())

        self.actions = np.zeros(self.statesize)

        #print(f"State: {self.state}, Goal: {self.goal}")
        return self.state, self.goal, self.error, self.done

    
    def step(self, velocity):
        self.t += 1
        self.actions = np.clip(velocity, -1,1) * self.maxspeed   # smoothen actions so that agent explores the entire arena. From Foster et al. 2000
        newstate = self.state.copy() + self.actions  # update state with action velocity

        self.track.append(self.state.copy())

        # check if new state crosses boundary
        if (newstate > self.maxsize).any() or (newstate < self.minsize).any():
            newstate = self.state.copy()
            self.actions = np.zeros(self.statesize)

        # check if new state crosses obstacles if initalized
        if self.obstacles:
            if  -0.6 < newstate[0] < -0.3 and  0.25 < newstate[1] < 1:  # top left obs
                newstate = self.state.copy()
                self.actions = np.zeros(self.statesize)
            
            if  -0.6 < newstate[0] < -0.3 and  -1 < newstate[1] < -0.25:  # bottom left obs
                newstate = self.state.copy()
                self.actions = np.zeros(self.statesize)
            
            if  0.3 < newstate[0] < 0.6 and  0.25 < newstate[1] < 1:  # top right obs
                newstate = self.state.copy()
                self.actions = np.zeros(self.statesize)
            
            if  0.3 < newstate[0] < 0.6 and  -1 < newstate[1] < -0.25:  # top right obs
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
        #plt.grid()
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
        
