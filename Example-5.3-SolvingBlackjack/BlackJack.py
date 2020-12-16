import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tqdm import tqdm


def stick(cards, i):
    value = cards[i] % 13

    if value < 10:
        value += 1
    else:
        value = 10
    i += 1
    return value


def game():
    cards = np.random.permutation(52)

    cards_player = []
    cards_dealer = []
    i = 0

    cards_dealer.append(stick(cards, i))
    i += 1
    cards_player.append(stick(cards, i))
    i += 1
    cards_dealer.append(stick(cards, i))
    i += 1
    cards_player.append(stick(cards, i))
    i += 1

    flag = 0
    index = 0
    if cards_player[0] == 1:
        cards_player[0] = 11
        index = 0
        flag = 1
    else:
        if cards_player[1] == 1:
            cards_player[1] = 11
            index = 1
            flag = 1

    while sum(cards_player) < 12:
        cards_player.append(stick(cards, i))
        i += 1
        if flag == 0 and cards_player[-1] == 1:
            index = len(cards_player) - 1
            cards_player[-1] = 11
            flag = 1

    return cards_player, cards_dealer, i, cards, flag, index


def maxposition(a):
    position = 0
    max = a[0]
    for i in range(len(a) - 1):
        if a[i + 1] > max:
            position = i + 1
            max = a[i + 1]
    return position


def action(q, s, epsilon):
    if np.random.uniform(0, 1, 1) < 1 - epsilon:
        p = maxposition(q[int(s[0, 0]), int(s[0, 1]), int(s[0, 2]), :])

    else:
        rand = np.random.uniform(0, 1, 1)
        if rand < 0.5:
            p = 0
        else:
            p = 1
    return p


def epsilonValue(gle, epsilon, epsilons, s):
    if gle == True:
        return 1 / epsilons[int(s[0, 0]), int(s[0, 1]), int(s[0, 2])]
    return epsilon


def playSarsa(gle):
    q = np.zeros((2, 10, 10, 2))
    iteration = 1000000
    gamma = 0.9
    j = 0
    epsilons = np.zeros((2, 10, 10))
    epsilon = 0.1
    alpha = 0.5

    rewards = []
    for i in tqdm(range(iteration)):
        s = np.zeros((1, 3))
        cards_player, cards_dealer, j, cards, flag, index = game()
        if flag == 1 and sum(cards_player) > 21:
            flag = 0
            cards_player[index] = 1
        s[0, 0] = flag
        s[0, 1] = cards_dealer[0] - 1
        s[0, 2] = sum(cards_player) - 12
        olds = np.array(s)

        epsilons[int(s[0, 0]), int(s[0, 1]), int(s[0, 2])] += 1
        p = action(q, s, epsilonValue(gle, epsilon, epsilons, s))

        while True:
            if p == 0:
                cards_player.append(stick(cards, j))
                j += 1
                if flag == 1:
                    if sum(cards_player) > 21:
                        flag = -1
                        cards_player[index] = 1
                        s[0, 0] = 0
                        s[0, 1] = cards_dealer[0] - 1
                        s[0, 2] = sum(cards_player) - 12
                    else:
                        s[0, 0] = 1
                        s[0, 1] = cards_dealer[0] - 1
                        s[0, 2] = sum(cards_player) - 12
                elif flag != 1:
                    if sum(cards_player) > 21:
                        reward = -1
                        q[int(s[0, 0]), int(s[0, 1]), int(s[0, 2]), p] = q[int(s[0, 0]), int(s[0, 1]), int(
                            s[0, 2]), p] + alpha * (reward - q[int(s[0, 0]), int(s[0, 1]), int(s[0, 2]), p])
                        break
                    else:
                        s[0, 0] = 0
                        s[0, 1] = cards_dealer[0] - 1
                        s[0, 2] = sum(cards_player) - 12

                epsilons[int(s[0, 0]), int(s[0, 1]), int(s[0, 2])] += 1
                p_prim = action(q, s, epsilonValue(gle, epsilon, epsilons, s))

                q[int(olds[0, 0]), int(olds[0, 1]), int(olds[0, 2]), p] = q[int(olds[0, 0]), int(olds[0, 1]), int(
                    olds[0, 2]), p] + \
                                                                          alpha * (q[int(s[0, 0]), int(s[0, 1]), int(
                    s[0, 2]), p_prim] - q[int(olds[0, 0]), int(olds[0, 1]), int(olds[0, 2]), p])

                olds = np.array(s)
                p = p_prim

            else:
                flagD = 0
                indexD = 0
                if cards_dealer[0] == 1:
                    cards_dealer[0] = 11
                    indexD = 0
                    flagD = 1
                else:
                    if cards_dealer[1] == 1:
                        cards_dealer[1] = 11
                        indexD = 1
                        flagD = 1
                while sum(cards_player) >= sum(cards_dealer):
                    cards_dealer.append(stick(cards, j))
                    j += 1
                    if flagD == 0 and cards_dealer[-1] == 1:
                        cards_dealer[-1] = 11
                        indexD = len(cards_dealer) - 1
                        flagD = 1

                    if flagD == 1 and sum(cards_dealer) > 21:
                        cards_dealer[indexD] = 1
                        flagD = -1

                reward = 0
                if sum(cards_player) > 21:
                    reward = -1
                elif sum(cards_dealer) > 21:
                    reward = 1
                elif sum(cards_dealer) < sum(cards_player):
                    reward = 1
                elif sum(cards_dealer) > sum(cards_player):
                    reward = -1

                q[int(s[0, 0]), int(s[0, 1]), int(s[0, 2]), 1] = q[int(s[0, 0]), int(s[0, 1]), int(s[0, 2]), 1] + \
                                                                 alpha * (reward - q[
                    int(s[0, 0]), int(s[0, 1]), int(s[0, 2]), 1])
                break

    return q


def playQLearning(gle):
    q = np.zeros((2, 10, 10, 2))
    iteration = 1000000
    gamma = 0.9
    j = 0
    epsilons = np.zeros((2, 10, 10))
    epsilon = 0.1
    alpha = 0.5

    rewards = []
    for i in tqdm(range(iteration)):
        s = np.zeros((1, 3))
        cards_player, cards_dealer, j, cards, flag, index = game()
        if flag == 1 and sum(cards_player) > 21:
            flag = 0
            cards_player[index] = 1
        s[0, 0] = flag
        s[0, 1] = cards_dealer[0] - 1
        s[0, 2] = sum(cards_player) - 12
        olds = np.array(s)

        epsilons[int(s[0, 0]), int(s[0, 1]), int(s[0, 2])] += 1
        p = action(q, s, epsilonValue(gle, epsilon, epsilons, s))

        while True:
            if p == 0:
                cards_player.append(stick(cards, j))
                j += 1
                if flag == 1:
                    if sum(cards_player) > 21:
                        flag = -1
                        cards_player[index] = 1
                        s[0, 0] = 0
                        s[0, 1] = cards_dealer[0] - 1
                        s[0, 2] = sum(cards_player) - 12
                    else:
                        s[0, 0] = 1
                        s[0, 1] = cards_dealer[0] - 1
                        s[0, 2] = sum(cards_player) - 12
                elif flag != 1:
                    if sum(cards_player) > 21:
                        reward = -1
                        q[int(s[0, 0]), int(s[0, 1]), int(s[0, 2]), p] = q[int(s[0, 0]), int(s[0, 1]), int(
                            s[0, 2]), p] + alpha * (reward - q[int(s[0, 0]), int(s[0, 1]), int(s[0, 2]), p])
                        break
                    else:
                        s[0, 0] = 0
                        s[0, 1] = cards_dealer[0] - 1
                        s[0, 2] = sum(cards_player) - 12

                epsilons[int(s[0, 0]), int(s[0, 1]), int(s[0, 2])] += 1
                p_prim = action(q, s, epsilonValue(gle, epsilon, epsilons, s))

                q[int(olds[0, 0]), int(olds[0, 1]), int(olds[0, 2]), p] = q[int(olds[0, 0]), int(olds[0, 1]), int(
                    olds[0, 2]), p] + \
                                                                          alpha * (max(
                    q[int(s[0, 0]), int(s[0, 1]), int(s[0, 2]), :]) - q[int(olds[0, 0]), int(olds[0, 1]), int(
                    olds[0, 2]), p])

                olds = np.array(s)
                p = p_prim

            else:
                flagD = 0
                indexD = 0
                if cards_dealer[0] == 1:
                    cards_dealer[0] = 11
                    indexD = 0
                    flagD = 1
                else:
                    if cards_dealer[1] == 1:
                        cards_dealer[1] = 11
                        indexD = 1
                        flagD = 1
                while sum(cards_player) >= sum(cards_dealer):
                    cards_dealer.append(stick(cards, j))
                    j += 1
                    if flagD == 0 and cards_dealer[-1] == 1:
                        cards_dealer[-1] = 11
                        indexD = len(cards_dealer) - 1
                        flagD = 1

                    if flagD == 1 and sum(cards_dealer) > 21:
                        cards_dealer[indexD] = 1
                        flagD = -1

                reward = 0
                if sum(cards_player) > 21:
                    reward = -1
                elif sum(cards_dealer) > 21:
                    reward = 1
                elif sum(cards_dealer) < sum(cards_player):
                    reward = 1
                elif sum(cards_dealer) > sum(cards_player):
                    reward = -1

                q[int(s[0, 0]), int(s[0, 1]), int(s[0, 2]), 1] = q[int(s[0, 0]), int(s[0, 1]), int(s[0, 2]), 1] + \
                                                                 alpha * (reward - q[
                    int(s[0, 0]), int(s[0, 1]), int(s[0, 2]), 1])
                break

    return q


def playQLearning(gle):
    q = np.zeros((2, 10, 10, 2))
    iteration = 1000000
    gamma = 0.9
    j = 0
    epsilons = np.zeros((2, 10, 10))
    epsilon = 0.1
    alpha = 0.5

    i = 0
    rewards=[]
    while i < iteration:
        s = np.zeros((1, 3))
        cards_player, cards_dealer, j, cards,flag,index = game()
        if flag == 1 and sum(cards_player)>21:
            flag = 0
            cards_player[index] = 1
        s[0,0] = flag
        s[0,1] = cards_dealer[0] - 1
        s[0,2] = sum(cards_player) - 12
        olds = np.array(s)

        epsilons[int(s[0, 0]), int(s[0, 1]), int(s[0, 2])] += 1
        p = action(q, s, epsilonValue(gle, epsilon,epsilons,s))

        while True :
            if p == 0:
                cards_player.append(stick(cards, j))
                j += 1
                if flag == 1:
                    if sum(cards_player) > 21:
                        flag = -1
                        cards_player[index] = 1
                        s[0,0] = 0
                        s[0,1] = cards_dealer[0] - 1
                        s[0, 2] = sum(cards_player) - 12
                    else:
                        s[0, 0] = 1
                        s[0, 1] = cards_dealer[0] - 1
                        s[0, 2] = sum(cards_player) - 12
                elif flag != 1:
                    if sum(cards_player) > 21:
                        reward = -1
                        q[int(s[0, 0]), int(s[0, 1]), int(s[0, 2]), p] = q[int(s[0, 0]), int(s[0, 1]), int(
                            s[0, 2]), p] + alpha * (reward - q[int(s[0, 0]), int(s[0, 1]), int(s[0, 2]), p])
                        break
                    else:
                        s[0, 0] = 0
                        s[0, 1] = cards_dealer[0] - 1
                        s[0, 2] = sum(cards_player) - 12

                epsilons[int(s[0, 0]), int(s[0, 1]), int(s[0, 2])] += 1
                p_prim = action(q, s, epsilonValue(gle, epsilon,epsilons,s))

                q[int(olds[0, 0]), int(olds[0, 1]), int(olds[0, 2]),p] = q[int(olds[0, 0]), int(olds[0, 1]), int(olds[0, 2]),p] + \
                                                  alpha*(max(q[int(s[0, 0]), int(s[0, 1]), int(s[0, 2]),:]) - q[int(olds[0, 0]), int(olds[0, 1]), int(olds[0, 2]),p])

                olds = np.array(s)
                p = p_prim

            else:
                flagD = 0
                indexD = 0
                if cards_dealer[0] == 1:
                    cards_dealer[0] = 11
                    indexD = 0
                    flagD = 1
                else:
                    if cards_dealer[1] == 1:
                        cards_dealer[1] = 11
                        indexD = 1
                        flagD = 1
                while sum(cards_player)>=sum(cards_dealer):
                    cards_dealer.append(stick(cards,j))
                    j += 1
                    if flagD == 0 and cards_dealer[-1] == 1:
                        cards_dealer[-1] = 11
                        indexD = len(cards_dealer) - 1
                        flagD = 1

                    if flagD == 1 and sum(cards_dealer)>21:
                        cards_dealer[indexD] = 1
                        flagD = -1

                reward = 0
                if sum(cards_player) > 21:
                    reward = -1
                elif sum(cards_dealer) > 21:
                    reward = 1
                elif sum(cards_dealer) < sum(cards_player):
                    reward = 1
                elif sum(cards_dealer) > sum(cards_player):
                    reward = -1

                q[int(s[0,0]),int(s[0,1]),int(s[0,2]), 1] = q[int(s[0,0]),int(s[0,1]),int(s[0,2]), 1] + \
                                                         alpha * (reward  - q[int(s[0,0]),int(s[0,1]),int(s[0,2]), 1])
                break

        i += 1

    return q

q = playSarsa(True)

bound0 = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        if q[0,i,j,0] < q[0,i,j,1]:
            bound0[9-j,i] = 1

bound1 = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        if q[1,i,j,0] < q[1,i,j,1]:
            bound1[9-j,i] = 1

print(bound0)
print(bound1)