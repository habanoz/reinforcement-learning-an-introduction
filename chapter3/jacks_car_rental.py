import time
import numpy as np
from scipy.stats import poisson

LOC2_EXPECTED_RETURN = 2
LOC2_EXPECTED_REQUEST = 4
LOC1_EXPECTED_RETURN = 3
LOC1_EXPECTED_REQUEST = 3
MAX_EXPECTED_UPDATE = 11

CAR_MOVE_COST = 2
RENT_PRICE = 10
MAX_CARS = 20
MAX_CARS_MOVED = 5
GAMMA = 0.9
PMF_MATRIX_MAP = {expected: [poisson.pmf(occurred, expected) for occurred in range(MAX_CARS + 1)]
                  for expected in [2, 3, 4]}
USE_EXPECTED_RETURNS = True


class State:
    def __init__(self, nLoc1, nLoc2, terminated=False):
        self.nLoc1 = nLoc1
        self.nLoc2 = nLoc2
        self.terminated = terminated


class ProbableUpdate:
    def __init__(self, req1, preq1, req2, preq2, ret1, pret1, ret2, pret2):
        self.req1 = req1
        self.preq1 = preq1

        self.req2 = req2
        self.preq2 = preq2

        self.ret1 = ret1
        self.pret1 = pret1

        self.ret2 = ret2
        self.pret2 = pret2


def step(action, s, V):
    """
    :param action: negative values for moving to location 1
    :param u tuple (req1, req2, ret1, ret2)
    :param s: current state, not modified
    :return: tuple of new state and reward
    """

    action_return = -abs(action) * CAR_MOVE_COST

    nLoc1 = s.nLoc1 - action
    nLoc2 = s.nLoc2 + action

    for u in states_updates(nLoc1, nLoc2):
        req1, req2, ret1, ret2 = min(u.req1, nLoc1), min(u.req2, nLoc2), u.ret1, u.ret2
        prob = u.preq1 * u.preq2 * u.pret1 * u.pret2

        spNLoc1 = min(nLoc1 - req1 + ret1, MAX_CARS)
        spNLoc2 = min(nLoc2 - req2 + ret2, MAX_CARS)

        reward = (req1 + req2) * RENT_PRICE + GAMMA * V[spNLoc1, spNLoc2]
        action_return += prob * reward

    return action_return


def states():
    for nLoc1 in range(MAX_CARS + 1):
        for nLoc2 in range(MAX_CARS + 1):
            yield State(nLoc1, nLoc2)


def states_updates(nLoc1, nLoc2):
    for req1 in range(MAX_EXPECTED_UPDATE):
        for req2 in range(MAX_EXPECTED_UPDATE):
            if USE_EXPECTED_RETURNS:
                yield ProbableUpdate(req1, PMF_MATRIX_MAP.get(LOC1_EXPECTED_REQUEST)[req1],
                                     req2, PMF_MATRIX_MAP.get(LOC2_EXPECTED_REQUEST)[req2],
                                     LOC1_EXPECTED_RETURN, 1.0,
                                     LOC2_EXPECTED_RETURN, 1.0
                                     )
            else:
                for ret1 in range(MAX_EXPECTED_UPDATE):
                    for ret2 in range(MAX_EXPECTED_UPDATE):
                        yield ProbableUpdate(req1, PMF_MATRIX_MAP.get(LOC1_EXPECTED_REQUEST)[req1],
                                             req2, PMF_MATRIX_MAP.get(LOC2_EXPECTED_REQUEST)[req2],
                                             ret1, PMF_MATRIX_MAP.get(LOC1_EXPECTED_RETURN)[ret1],
                                             ret2, PMF_MATRIX_MAP.get(LOC2_EXPECTED_RETURN)[ret2]
                                             )


def policy_evaluation(V, pi):
    print("policy_evaluation started")

    theta = 0.0001
    delta = 1
    iteration = 0

    while delta > theta:
        delta = 0
        iteration += 1
        istart = time.time()

        for s in states():
            v = V[s.nLoc1, s.nLoc2]

            a = pi[s.nLoc1, s.nLoc2]
            action_return = step(a, s, V)
            V[s.nLoc1, s.nLoc2] = action_return

            delta = max(abs(action_return - v), delta)

        # if iteration % 10 == 0:
        print("policy_evaluation iteration {}, max delta='{}' in {} seconds"
              .format(iteration, delta, time.time() - istart))


def policy_improvement(V, pi):
    start = time.time()
    print("policy_improvement started")

    policy_stable = True

    for s in states():
        old_action = pi[s.nLoc1, s.nLoc2]
        actions = np.arange(-s.nLoc2, s.nLoc1 + 1)  # negative actions goes to location 1
        action_returns = np.empty_like(actions)

        for a in range(len(actions)):
            action_return = step(a, s, V)
            action_returns[a] = action_return

        pi[s.nLoc1, s.nLoc2] = actions[np.argmax(action_returns)]

        if pi[s.nLoc1, s.nLoc2] != old_action: policy_stable = False

    print("policy_improvement completed in {} seconds".format(time.time() - start))

    return policy_stable


def policy_iteration():
    V = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    pi = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=int)

    policy_stable = False
    iteration = 0
    while not policy_stable:
        policy_evaluation(V, pi)
        policy_stable = policy_improvement(V, pi)
        iteration += 1
        print("Policy iteration {} completed".format(iteration))

    np.savetxt("v.txt", V)
    np.savetxt("pi.txt", pi)


if __name__ == '__main__':
    policy_iteration()
