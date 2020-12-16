import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tqdm import tqdm


def hit(cards,i):
    '''

    :param cards: shuffled cards
    :param i: index of first card have not been showed yet in shuffled cards
    :return: value of first card have not showed yet
    '''
    value = cards[i]%13

    if value <10:
        value += 1
    else:
        value = 10
    return value

vs = np.zeros((2,10,10))
ns = np.zeros((2,10,10))
iteration = 1000000
gamma = 0.9

for i in tqdm(range(iteration)):
    #shuffling cards
    cards=np.random.permutation(52)


    #intialization
    s = 0
    cards_player =[]
    cards_dealer =[]
    i = 0

    #dealing two cards for each player(dealer and player)
    cards_dealer.append(hit(cards,i))
    i += 1
    cards_player.append(hit(cards, i))
    i += 1
    cards_dealer.append(hit(cards,i))
    i += 1
    cards_player.append(hit(cards, i))
    i += 1

    #checking usable ace between cards of player and set flag and index varaible if player has any.
    flag = 0
    index = 0
    state = []
    if cards_player[0] == 1:
        cards_player[0] = 11
        index = 0
        flag = 1
    else:
        if cards_player[1] == 1:
            cards_player[1] = 11
            index = 1
            flag = 1

    #append first state
    state.append([sum(cards_player),flag])

    #hit card until reach sum of player's cards is over 19 as policy is defined
    while sum(cards_player) < 20 :
        cards_player.append(hit(cards, i))
        i += 1
        #checking for ace if player does not have ace
        if flag == 0 and cards_player[-1] == 1:
            index = len(cards_player) - 1
            cards_player [-1] = 11
            flag = 1
        # checking for usable ace if player's sum of cards is over 21
        if flag == 1 and sum(cards_player) > 21:
            cards_player[index] = 1
            flag = -1
        #add new state after new action
        if flag == 0 or flag == -1:
          state.append([sum(cards_player),0])
        else:
          state.append([sum(cards_player),1])



    # checking usable ace between cards of dealer and set flag and index varaible if player has any.
    flag = 0
    if cards_dealer[0] == 1:
        cards_dealer[0] = 11
        index = 0
        flag = 1
    else:
        if cards_dealer[1] == 1:
            cards_dealer[1] = 11
            index = 1
            flag = 1

    while sum(cards_dealer) < sum(cards_player) and sum(cards_player) < 22 and sum(cards_dealer) < 22:
        cards_dealer.append(hit(cards, i))
        i += 1
        # checking for ace if dealer does not have ace
        if flag == 0 and cards_dealer[-1] == 1:
            index = len(cards_dealer) - 1
            cards_dealer [-1] = 11
            flag = 1
        # checking for usable ace if player's sum of cards is over 21
        if flag == 1 and sum(cards_dealer) > 21:
            cards_dealer[index] = 1
            flag = -1


    #calculate reward
    reward = 0
    if sum(cards_player) > 21:
        reward = -1
    elif sum(cards_dealer) > 21:
        reward = 1
    elif sum(cards_player) > sum(cards_dealer):
        reward = 1
    elif sum(cards_player) == sum(cards_dealer):
        reward = 0
    else:
        reward = -1

    # caculate value of dealer's first card
    if cards_dealer[0] == 11:
        index2 = 0
    else:
        index2 = cards_dealer[0] - 1



    #calculate value states of problem using first visit algorithm
    for i in range(len(state)):
      sumPlayer, flagAce = state[i]
      # control state regarding valid or not
      if sumPlayer > 11 and sumPlayer < 22:
        flag = 0
        #control state regarding it is observed in past or not
        for j in range(i):
          sumPlayerPast, flagAcePast = state[j]
          if sumPlayer == sumPlayerPast:
            if flagAce == flagAcePast:
              flag = 1
              break
        #if state is not observed and it is valid update counter of state and value state
        if flag == 0:
          index1 = flagAce
          ns[index1,index2,sumPlayer - 12] += 1
          Gt = reward * pow(gamma,len(state) - i - 1)
          vs [index1,index2,sumPlayer - 12] += (1/ns[index1,index2,sumPlayer - 12])*(Gt - vs [index1,index2,sumPlayer - 12])
          vs [index1,index2,sumPlayer - 12] += reward

#divide summation values of states to counter of states elemnt wise and print it
vs = vs/ns
print(vs[0,:,:])
print(vs[1,:,:])


#plot value state
x = np.linspace(12, 21, 10)
y = np.linspace(1, 10, 10)
X,Y = np.meshgrid(x,y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, vs[1,:,:], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none');
ax.set_xlabel('Sum of player cards')
ax.set_ylabel('first card of dealer')
ax.set_zlabel('Vs without usable ace')
ax.view_init(45,210)
plt.show()

x = np.linspace(12, 21, 10)
y = np.linspace(1, 10, 10)
X,Y = np.meshgrid(x,y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, vs[1,:,:], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none');

ax.set_xlabel('Sum of player cards')
ax.set_ylabel('first card of dealer')
ax.set_zlabel('Vs with usable ace')
ax.view_init(45,210)
plt.show()
