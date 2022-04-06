# vary 1: S & B , multi update, 52 card
# vary 2: Tsitsili (MIT), uniform initial state, 1 update per episode
# vary 3: uniform initial state, multi update

import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
import datetime

def getStateUniform():
    sumValue = np.random.randint(12, 22)
    usableAce = bool(np.random.randint(0, 2))
    cards = []
    if usableAce: # Ace count as 11
        cards.append(1)
        cards.append(sumValue - 11)
    else:
        cards.append(sumValue)
    return cards

class Player(object):
    def __init__(self, dealersCard):
        self.cards = [] # each card is a number 1-10
        self.dealersCard = dealersCard
    def CheckAceUsable(self):
        # check whether it's possible to use Ace
        baseSum = sum(self.cards)
        if 1 in self.cards and baseSum <= 11:
            return True
        else:
            return False
    def GetValue(self):
        # get the current value, if player has an usable Ace, it will be count as 11 here
        currentSum = 0
        hasAce = False

        for card in self.cards:
            if card == 1:
                hasAce = True
                currentSum += 1
            else:
                currentSum += card

        # now the currentSum is the value when we do not use an Ace (all Ace count as 1)
        if hasAce and currentSum <= 11:
            # crucial here, since using an Ace will give us 10 more points, then if we have > 11 points when counting all Ace
            # as 1, then we simply can't use an Ace.
            currentSum += 10
        return currentSum
    def Bust(self):
        return self.GetValue() > 21
    def AddCard(self, card):
        self.cards.append(card)
    def GetState(self):
        return (self.GetValue(), self.CheckAceUsable(), self.dealersCard)
    def ShouldHit(self, policyMap):
        return policyMap[self.GetState()]

class Dealer(object):
    def __init__(self, cards):
        self.cards = cards

    def AddCard(self, card):
        self.cards.append(card)

    def Bust(self):
        return self.GetValue() > 21

    def GetValue(self):
        currentSum = 0
        hasAce = 0

        for card in self.cards:
            if card == 1:
                hasAce = True
                currentSum += 1
            else:
                currentSum += card

        # now the currentSum is the value when we do not use an Ace (all Ace count as 1)
        if hasAce and currentSum <= 11:
            # crucial here, since using an Ace will give us 10 more points, then if we have > 11 points when counting all Ace
            # as 1, then we simply can't use an Ace.
            currentSum += 10
        return currentSum

    def ShouldHit(self):
        if self.GetValue() >= 17:
            return False
        else:
            return True

class StateActionInfo(object):
    def __init__(self):
        self.stateActionPairs = [ ]
        self.stateActionMap = set()

    def AddPair(self, pair):
        if pair in self.stateActionMap:
            return

        self.stateActionPairs.append(pair)
        self.stateActionMap.add(pair)

def EvaluateAndImprovePolicy(actionValueMap, policyMap, returns, stateActionPairs, reward, episode):
    QUpdate[seed].append(0)
    for pair in stateActionPairs:
        returns[pair] += 1
        QUpdate[seed][episode] += abs(((reward - actionValueMap[pair]) / returns[pair]))
        actionValueMap[pair] = actionValueMap[pair] + ((reward - actionValueMap[pair]) / returns[pair])

        state = pair[0]
        shouldHit = False

        if actionValueMap[(state, True)] >= actionValueMap[(state, False)]:
            shouldHit = True

        policyMap[state] = shouldHit
    QUpdate[seed][episode] /=len(stateActionPairs)

def GenerateCard():
    card = np.random.randint(1, 14)

    if card > 9:
        return 10
    else:
        return card

def GetNewCard():
    # will return a card, J, Q, K will be returned as 10
    card = np.random.randint(1, 14)
    if card > 10:
        card = 10
    return card

def GenerateEpisode(actionValueMap, policyMap, returns, episode, uniform_initial = False, initial_update = False):
    # 52 cards initial inversion
    if not uniform_initial:
        dealersCard1 = GetNewCard()
        dealer = Dealer([dealersCard1])
        player = Player(dealersCard1)
        # get 2 inital cards
        player.AddCard(GetNewCard())
        player.AddCard(GetNewCard())
        # if value is < 11, then keep getting cards
        while player.GetValue() < 11: # keep getting cards until we have at least a value of 11
            player.AddCard(GetNewCard())
    # uniform initial version
    else: # TODO worry about this part later... might also have problems
        dealersCard1 = GetNewCard()
        dealer = Dealer([dealersCard1])
        player = Player(dealersCard1)

        # player can have ace/no-ace and value is 11-21

        # has usable ace: player must have ace, base sum <= 11

        # no usable ace: 1) player has base card numbers of > 11: impossible to use ace
        # 2) player does not have ace yet: player has a base of 11, but no ace.

        # TODO here how do we simply set up player's value?
        sum = np.random.randint(12, 22)
        useableAce = bool(np.random.randint(0, 2))
        if useableAce:
            player.AddCard(1)
            player.AddCard(sum - 11)
        else:
            player.AddCard(sum // 3)
            player.AddCard(sum // 3)
            player.AddCard(sum - (sum // 3) * 2)
        # dealersCard1 = np.random.randint(1, 11)

    stateActionInfo = StateActionInfo() # used to store s,a pairs in the episode
    shouldHit = bool(np.random.randint(0, 2)) # initial action
    stateActionInfo.AddPair((player.GetState(), shouldHit))

    if shouldHit:
        player.AddCard(GenerateCard())
        while not player.Bust() and player.ShouldHit(policyMap):
            if not initial_update:
                stateActionInfo.AddPair((player.GetState(), True))
            player.AddCard(GenerateCard())

    if player.Bust():
        EvaluateAndImprovePolicy(actionValueMap, policyMap, returns, stateActionInfo.stateActionPairs, -1, episode)
        return

    if not initial_update:
        stateActionInfo.AddPair((player.GetState(), False))
    dealer.AddCard(GenerateCard()) # dealer get 2nd card (mandatory by rule)

    while not dealer.Bust() and dealer.ShouldHit():
        dealer.cards.append(GenerateCard())

    if dealer.Bust() or dealer.GetValue() < player.GetValue():
        EvaluateAndImprovePolicy(actionValueMap, policyMap, returns, stateActionInfo.stateActionPairs, 1, episode)
    elif dealer.GetValue() > player.GetValue():
        EvaluateAndImprovePolicy(actionValueMap, policyMap, returns, stateActionInfo.stateActionPairs, -1, episode)
    else:
        EvaluateAndImprovePolicy(actionValueMap, policyMap, returns, stateActionInfo.stateActionPairs, 0, episode)

def EvaluatePerformance(policyMap, game_number = 10000):
    performance = []
    for i in range(game_number):
        dealersCard1 = GetNewCard()
        dealer_test = Dealer([dealersCard1])
        player_test = Player(dealersCard1)
        player_test.AddCard(GetNewCard())
        player_test.AddCard(GetNewCard())
        while player_test.GetValue() < 11: # keep getting cards until we have at least a value of 11
            player_test.AddCard(GetNewCard())

        while not player_test.Bust() and player_test.ShouldHit(policyMap):
            player_test.AddCard(GenerateCard())

        if player_test.Bust():
            performance.append(-1)
        else:
            while not dealer_test.Bust() and dealer_test.ShouldHit():
                dealer_test.cards.append(GenerateCard())
            if dealer_test.Bust() or dealer_test.GetValue() < player_test.GetValue():
                performance.append(1)
            elif dealer_test.GetValue() > player_test.GetValue():
                performance.append(-1)
            else:
                performance.append(0)

    return sum(performance)/game_number

def CheckPolicyConverge(ground_true_policy,policyMap):
    count = 0
    for usableAce in range(2):
        for playerSum in range(11, 22):
            for dealersCard in range(1, 11):
                playerState = (playerSum, bool(usableAce), dealersCard)
                if policyMap[playerState] == ground_true_policy[playerState]:
                    count += 1

    return count

# def CheckAVMap(previousAVMap,actionValueMap,converge_epsilon):
#     for usableAce in range(2):
#         for playerSum in range(11, 22):
#             for dealersCard in range(1, 11):
#                 for hit in range(0,2):
#                     playerState = (playerSum, bool(usableAce), dealersCard)
#                     # print(abs(previousAVMap[(playerState, bool(hit))] - actionValueMap[(playerState, bool(hit))]))
#                     if abs(previousAVMap[(playerState, bool(hit))] - actionValueMap[(playerState, bool(hit))]) > converge_epsilon:
#                         return False
#
#     return True

def PerformMonteCarloES(seed, uniform_initial = False, multi_update = True):
    actionValueMap = { }
    policyMap = { } # map playerState to True or False, True is hit?
    returns = { }
    ground_true_policy = {}
    # init
    for usableAce in range(2):
        for playerSum in range(11, 22):
            for dealersCard in range(1, 11):
                playerState = (playerSum, bool(usableAce), dealersCard)
                actionValueMap[(playerState, False)] = 0 # Q value
                actionValueMap[(playerState, True)] = 0
                returns[(playerState, False)] = 0 # number of returns?
                returns[(playerState, True)] = 0

                if playerSum == 20 or playerSum == 21:
                    policyMap[playerState] = False
                else:
                    policyMap[playerState] = True

                if usableAce:
                    if dealersCard < 2 or dealersCard > 8:
                        if playerSum > 18:
                            ground_true_policy[playerState] = False
                        else:
                            ground_true_policy[playerState] = True
                    else:
                        if playerSum > 17:
                            ground_true_policy[playerState] = False
                        else:
                            ground_true_policy[playerState] = True
                else:
                    if dealersCard < 2 or dealersCard > 6:
                        if playerSum > 16:
                            ground_true_policy[playerState] = False
                        else:
                            ground_true_policy[playerState] = True
                    elif dealersCard < 4:
                        if playerSum > 12:
                            ground_true_policy[playerState] = False
                        else:
                            ground_true_policy[playerState] = True
                    else:
                        if playerSum > 11:
                            ground_true_policy[playerState] = False
                        else:
                            ground_true_policy[playerState] = True

    # print(ground_true_policy)

    n_episode = int(1e7)
    for i in range(n_episode):
        GenerateEpisode(actionValueMap, policyMap, returns, i, uniform_initial, not multi_update) # includes running an episode and change policy
        if (i+1) % 10 == 0 and i < int(1e6):
           score = EvaluatePerformance(policyMap, 1000)
           multiseed_performance[seed].append(score)
        if (i+1) % 100000 == 0:
           print(i)
        number = CheckPolicyConverge(ground_true_policy,policyMap)
        policy_converge_number[seed].append(number)

    x11 = [ ]
    y11 = [ ]

    x12 = [ ]
    y12 = [ ]

    x21 = [ ]
    y21 = [ ]

    x22 = [ ]
    y22 = [ ]

    # for every state, check what is the policy, either add a red dot or add a blue dot
    for playerState in policyMap:
        if playerState[1]: # if usable ace
            if policyMap[playerState]: # if policy is to hit
                x11.append(playerState[2] - 1) # playerState[2] is dealer card
                y11.append(playerState[0] - 11) # playerState[0] is player card
            else:
                x12.append(playerState[2] - 1)
                y12.append(playerState[0] - 11)
        else:
            if policyMap[playerState]:
                x21.append(playerState[2] - 1)
                y21.append(playerState[0] - 11)
            else:
                x22.append(playerState[2] - 1)
                y22.append(playerState[0] - 11)

    # TODO here we should print out the action values in a matrix
    # TODO make sure to test it before running stuff..
    for usableAce in range(2):
        print("usable:", usableAce)
        for playerSum in range(11, 22):
            line = ''
            for dealersCard in range(1, 11):
                playerState = (playerSum, bool(usableAce), dealersCard)
                line += '%d,%d:%.3f/%.3f\t' % (playerSum, dealersCard, actionValueMap[(playerState, True)], actionValueMap[(playerState, False)])
                # returns[(playerState, False)] = 0 # number of returns?
            print(line)
        print()

    # plot trained policy
    # plt.figure(0)
    # plt.title('With Usable Ace')
    # plt.scatter(x11, y11, color='red')
    # plt.scatter(x12, y12, color='blue')
    # plt.xticks(range(10), [ 'A', '2', '3', '4', '5', '6', '7', '8', '9', '10' ])
    # plt.yticks(range(11), [ '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21' ])
    #
    # plt.figure(1)
    # plt.title('Without Usable Ace')
    # plt.scatter(x21, y21, color='red')
    # plt.scatter(x22, y22, color='blue')
    # plt.xticks(range(10), [ 'A', '2', '3', '4', '5', '6', '7', '8', '9', '10' ])
    # plt.yticks(range(11), [ '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21' ])
    #
    # plt.figure(2)
    # plt.title('Performance')
    # x = np.arange(1000)
    # plt.plot(x, performance)
    #
    # plt.figure(3)
    # plt.title('PolicyConvergeEval')
    # x = np.arange(n_episode)
    # plt.plot(x, policy_converge_number)
    #
    # plt.show()

    # print('QConvergeTime:',QConvergeTime)

date = datetime.date.today()
n_seed = 5
for uniform_initialization in range(0,2):
    for multiple_update in range(0,2):
        uniform_initialization = bool(uniform_initialization)
        multiple_update = bool(multiple_update)

        multiseed_performance = [[] for seed in range(5)]
        policy_converge_number = [[] for seed in range(5)]
        QUpdate = [[] for seed in range(5)]
        for seed in range(n_seed):
            np.random.seed(seed)
            PerformMonteCarloES(seed, uniform_initialization, multiple_update) #seed, whether uniform initial, whether multi-update


        performanceFile = np.array(multiseed_performance)
        performanceFile.tofile('Performance_UI_%s_M_%s_%s.csv' % (uniform_initialization, multiple_update, date),sep=',')

        policyFile = np.array(policy_converge_number)
        policyFile.tofile('Policy_UI_%s_M_%s_%s.csv' % (uniform_initialization, multiple_update, date),sep=',')

        QFile = np.array(QUpdate)
        QFile.tofile('QUpdate_UI_%s_M_%s_%s.csv' % (uniform_initialization, multiple_update, date),sep=',')

