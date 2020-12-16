import numpy as np
import matplotlib.pyplot as plt


def Game():
    state = 3

    s = [3]

    while s[-1] != 0 and s[-1] != 6:
        if np.random.binomial(1, 0.5)>0.5:
            s.append(s[-1]+1)
        else:
            s.append(s[-1]-1)

    return s

def mc (vs,game,alpha):
    gamma = 1
    reward = 0
    if game[-1] == 6:
        reward = 1


    for i in range(len(game) - 1):
        vs[0,game[i] - 1] += alpha*(reward - vs[0,game[i] - 1])

    return vs

def td (vs, game , alpha):
    gamma = 1
    reward = 0
    if game[-1] == 6:

        reward = 1

    for i in range(len(game) - 2):
        vs[0, game[i] - 1] += alpha * (gamma*vs[0, game[i + 1] - 1] - vs[0, game[i] - 1])
    vs[0, game[-2] - 1] += alpha * (reward - vs[0, game[-2] - 1])

    return vs


games = []
for i in range(100):
    games.append(Game())



alpha = [0.01, 0.02, 0.03, 0.04]

rms = np.zeros((len(alpha),len(games)+1))
true_vs = np.zeros((1,5)) + 0.5
true_vs [0,0] = 1/6
true_vs [0,1] = 2/6
true_vs [0,3] = 4/6
true_vs [0,4] = 5/6

x = np.arange(len(games) + 1)
for i in range(len(alpha)):
    for k in range(100):
        vs = np.zeros((1, 5)) + 0.5
        rms[i, 0] += np.sqrt(np.mean((vs - true_vs) ** 2))
        for j in range(len(games)):
            vs = mc(vs, Game(), alpha[i])
            rms[i,j + 1] += np.sqrt(np.mean((vs-true_vs)**2))
    plt.plot(x, rms[i,:]/100,label="MC  "+str(alpha[i]) )

alpha = [0.05, 0.1, 0.15]
rms_td = np.zeros((len(alpha),len(games)+1))
vs0 = np.zeros((4,5))
vs0[0,:]=np.zeros((1, 5)) + 0.5

for i in range(len(alpha)):


    for k in range(100):
        vs = np.zeros((1, 5)) + 0.5
        rms_td[i, 0] += np.sqrt(np.mean((vs - true_vs) ** 2))
        for j in range(len(games)):

            vs = td(vs, Game(), alpha[i])
            rms_td[i,j + 1] += np.sqrt(np.mean((vs-true_vs)**2))
            if alpha[i] == 0.1:
                if j == 0:
                    vs0[1, :] += vs[0,:]
                elif j == 9:
                    vs0[2, :] += vs[0,:]
                elif j == 99:
                    vs0[3, :] += vs[0,:]



    plt.plot(x, rms_td[i,:]/100, linestyle='dashed', label="TD  "+str(alpha[i]))
plt.legend()
plt.show()


x = np.arange(5) + 1
plt.plot(x, np.reshape(true_vs,(5,1)),marker='o', linestyle='dashed', label="True Values")
plt.plot(x, vs0[0,:],marker='o', linestyle='dashed', label="0")
plt.plot(x, vs0[1,:]/100,marker='o', linestyle='dashed', label="1")
plt.plot(x, vs0[2,:]/100,marker='o', linestyle='dashed', label="10")
plt.plot(x, vs0[3,:]/100,marker='o', linestyle='dashed', label="100")
plt.legend()
plt.show()
