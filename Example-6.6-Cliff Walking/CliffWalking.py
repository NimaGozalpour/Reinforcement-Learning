import numpy as np
import matplotlib.pyplot as plt

def game(state,action):

    if action == 'w':
        state[0,0] -= 1
    elif action == 'a':
        state[0, 1] -= 1
    elif action == 'd':
        state[0, 1] += 1
    else:
        state[0, 0] += 1



    if state[0,1] < 0:
        state[0, 1] = 0
    elif state[0,1] > 11:
        state[0, 1] = 11

    if state[0,0] < 0:
        state[0, 0] = 0
    elif state[0,0] > 3:
        state[0, 0] = 3


    return state


def playameManualy():
    s = np.zeros((1,2))
    snew = np.zeros((1,2))
    s[0,0] = 3
    flag = 1
    while flag == 1:
        a = input("enter command: ")
        if a == 'f':
            break
        olds = np.array(s)
        s = game(s,a)


        print(f)
        state = np.zeros((4, 12))
        state[int(olds[0,0]),int(olds[0,1])] = -1
        state[int(s[0, 0]), int(s[0, 1])] = 1
        print(state)




def maxposition(a):
    position = 0
    max = a[0]
    for i in range(len(a)-1):
        if a[i+1] > max:
            position = i + 1
            max = a[i+1]
    return position

def action(q,s,epsilon):
    if np.random.uniform(0,1,1) < 1-epsilon:
        p = maxposition(q[int(4*s[0,1]+s[0,0]),:])

    else:
        rand = np.random.uniform(0,1,1)
        if rand < 0.25:
            p = 0
        elif rand < 0.5:
            p = 1
        elif rand <0.75:
            p = 2
        else:
            p = 3
    return p

def epsilonValue(gle, epsilon,epsilons,s):
    if gle == True:
        return 1/epsilons[int(4*s[0,1]+s[0,0]),0]
    return epsilon
def playSarsa(gle):
    q = np.zeros((48, 4))
    actions = ['w','a','d','s']
    epsilons = np.ones((48,1))
    epsilon = 0.1
    alpha = 0.5

    i = 0
    rewards=[]
    while i < 550:
        s = np.zeros((1, 2))
        s[0, 0] = 3
        olds = np.array(s)
        p = action(q, s, epsilonValue(gle, epsilon,epsilons,s))
        epsilons[int(4 * s[0, 1] + s[0, 0]), 0] += 1
        a = actions[p]
        r = 0
        while s[0,0] != 3 or s[0,1] != 11 :
            s = game(s, a)

            reward = -1
            r +=reward
            p_prim = action(q, s, epsilonValue(gle, epsilon,epsilons,s))
            epsilons[int(4 * s[0, 1] + s[0, 0]), 0] += 1

            q[int(4*olds[0,1]+olds[0,0]),p] = q[int(4*olds[0,1]+olds[0,0]),p] + \
                                              alpha*(reward + q[int(4*s[0,1]+s[0,0]),p_prim] - q[int(4*olds[0,1]+olds[0,0]),p])

            if s[0, 0] == 3:
                if s[0, 1] > 0 and s[0, 1] < 11:
                    olds = np.array(s)
                    p = p_prim
                    a = actions[p]
                    s = np.zeros((1, 2))
                    s[0, 0] = 3
                    reward = -100
                    r += reward
                    p_prim = action(q, s, epsilonValue(gle, epsilon,epsilons,s))
                    epsilons[int(4 * s[0, 1] + s[0, 0]), 0] += 1
                    q[int(4 * olds[0, 1] + olds[0, 0]), p] = q[int(4 * olds[0, 1] + olds[0, 0]), p] + alpha * (
                            reward + q[int(4 * s[0, 1] + s[0, 0]), p_prim] - q[int(4 * olds[0, 1] + olds[0, 0]), p])
                    break

            olds = np.array(s)
            p = p_prim
            a = actions[p]

        i += 1
        rewards.append(r)

    return q,rewards

def playQLearning(gle):
    q = np.zeros((48, 4))
    actions = ['w','a','d','s']
    epsilons = np.ones((48,1))
    epsilon = 0.1
    alpha = 0.5

    i = 0
    rewards=[]
    while i < 550:
        s = np.zeros((1, 2))
        s[0, 0] = 3
        olds = np.array(s)
        p = action(q, s, epsilonValue(gle, epsilon,epsilons,s))
        epsilons[int(4 * s[0, 1] + s[0, 0]), 0] += 1
        a = actions[p]
        r = 0
        while s[0,0] != 3 or s[0,1] != 11 :
            s = game(s, a)

            reward = -1
            r +=reward
            p_prim = action(q, s, epsilonValue(gle, epsilon,epsilons,s))
            epsilons[int(4 * s[0, 1] + s[0, 0]), 0] += 1

            q[int(4*olds[0,1]+olds[0,0]),p] = q[int(4*olds[0,1]+olds[0,0]),p] + \
                                              alpha*(reward + max(q[int(4*s[0,1]+s[0,0]),:]) - q[int(4*olds[0,1]+olds[0,0]),p])

            if s[0, 0] == 3:
                if s[0, 1] > 0 and s[0, 1] < 11:
                    olds = np.array(s)
                    p = p_prim
                    a = actions[p]
                    s = np.zeros((1, 2))
                    s[0, 0] = 3
                    reward = -100
                    r += reward
                    p_prim = action(q, s, epsilonValue(gle, epsilon,epsilons,s))
                    epsilons[int(4 * s[0, 1] + s[0, 0]), 0] += 1
                    q[int(4 * olds[0, 1] + olds[0, 0]), p] = q[int(4 * olds[0, 1] + olds[0, 0]), p] + alpha * (
                            reward + max(q[int(4 * s[0, 1] + s[0, 0]), :]) - q[int(4 * olds[0, 1] + olds[0, 0]), p])
                    break

            olds = np.array(s)
            p = p_prim
            a = actions[p]

        i += 1
        rewards.append(r)

    return q,rewards

def bestRouteFinder(q):
    s = np.zeros((1, 2))
    s[0, 0] = 3

    state = np.zeros((4, 12))
    state [int(s[0, 0]),int(s[0, 1])] = 1
    actions = ['w','a','d','s']
    while s[0, 0] != 3 or s[0, 1] != 11:
        p = maxposition(q[int(4 * s[0, 1] + s[0, 0]), :])
        a = actions[p]
        s = game(s, a)
        state[int(s[0, 0]), int(s[0, 1])] = 1

    print(state)


delay = 50

q,reward=playSarsa(True)
smoothedReward =[]
for i in range(len(reward)-delay):
    smoothedReward.append(sum(reward[i:i+delay])/delay)

reward =smoothedReward
x = np.arange(len(reward))
plt.plot(x,reward, linestyle='dashed', label="Sarsa-Gle")
print("Best Route is finded by Sarsa-Gle")
bestRouteFinder(q)


q,reward=playSarsa(False)
smoothedReward =[]
for i in range(len(reward)-delay):
    smoothedReward.append(sum(reward[i:i+delay])/delay)

reward =smoothedReward
x = np.arange(len(reward))
plt.plot(x,reward, linestyle='dashed', label="Sarsa")
print("Best Route is finded by Sarsa")
bestRouteFinder(q)

q,reward=playQLearning(True)
smoothedReward =[]
for i in range(len(reward)-delay):
    smoothedReward.append(sum(reward[i:i+delay])/delay)

reward =smoothedReward
x = np.arange(len(reward))
plt.plot(x,reward,  label="Q-Learning-Gle")
print("Best Route is finded by Q-Learning-Gle")
bestRouteFinder(q)

q,reward=playQLearning(False)
smoothedReward =[]
for i in range(len(reward)-delay):
    smoothedReward.append(sum(reward[i:i+delay])/delay)

reward =smoothedReward
x = np.arange(len(reward))
plt.plot(x,reward,  label="Q-Learning")
plt.legend()
plt.show()

print("Best Route is finded by Q-Learning")
bestRouteFinder(q)