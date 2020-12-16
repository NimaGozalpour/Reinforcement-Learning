import numpy as np
import matplotlib.pyplot as plt







def game(state,action):

    w = wind(state)
    if action == 'w':
        state[0,0] -= 1
    elif action == 'a':
        state[0, 1] -= 1
    elif action == 'd':
        state[0, 1] += 1
    else:
        state[0, 0] += 1
    state[0, 0] -= w

    if state[0,0] < 0:
        state[0, 0] = 0
    elif state[0,0] > 6:
        state[0, 0] = 6

    if state[0,1] < 0:
        state[0, 1] = 0
    elif state[0,1] > 9:
        state[0, 1] = 9

    return state

def wind(state):
    if state[0,1] <3:
        w = 0
    elif state[0,1]<6:
        w = 1
    elif state[0,1] <8:
        w = 2
    elif state[0,1]<9:
        w = 1
    else:
        w = 0
    return w



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


        print(s,olds)
        state = np.zeros((7, 10))
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
        p = maxposition(q[int(7*s[0,1]+s[0,0]),:])

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

def playTD():
    q = np.zeros((70, 4))
    actions = ['w','a','d','s']
    epsilons = np.ones((70,1))
    epsilon = 0.1
    alpha = 0.5

    i = 0
    j = 0
    epi_step=[]
    while i < 8000:
        s = np.zeros((1, 2))
        s[0, 0] = 3
        olds = np.array(s)
        p = action(q, s, 1/epsilons[int(7*s[0,1]+s[0,0]),0])
        epsilons[int(7 * s[0, 1] + s[0, 0]), 0] += 0.5
        a = actions[p]
        while s[0,0] != 3 or s[0,1] != 7 :
            s = game(s, a)
            reward = -1
            p_prim = action(q, s, 1/epsilons[int(7*s[0,1]+s[0,0]),0])
            epsilons[int(7 * s[0, 1] + s[0, 0]), 0] += 0.5

            q[int(7*olds[0,1]+olds[0,0]),p] = q[int(7*olds[0,1]+olds[0,0]),p] + alpha*(reward + q[int(7*s[0,1]+s[0,0]),p_prim] - q[int(7*olds[0,1]+olds[0,0]),p])

            olds = np.array(s)
            p = p_prim
            a = actions[p]
            i += 1
            epi_step.append(j)
        j += 1

    return q,epi_step

q,epi_step=playTD()

x = np.arange(len(epi_step))
plt.plot(x,epi_step, linestyle='dashed')
plt.show()


s = np.zeros((1, 2))
s[0, 0] = 3

state = np.zeros((7, 10))
state [int(s[0, 0]),int(s[0, 1])] = 1
actions = ['w','a','d','s']
while s[0, 0] != 3 or s[0, 1] != 7:
    p = maxposition(q[int(7 * s[0, 1] + s[0, 0]), :])
    a = actions[p]
    s = game(s, a)
    state[int(s[0, 0]), int(s[0, 1])] = 1

print(state)