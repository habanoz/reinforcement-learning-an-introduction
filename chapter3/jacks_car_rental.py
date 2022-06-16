import numpy
import numpy as np
from scipy.stats import poisson

LOC2_EXPECTED_RETURN = 4
LOC2_EXPECTED_REQUEST = 2
LOC1_EXPECTED_RETURN = 3
LOC1_EXPECTED_REQUEST = 3

CAR_MOVE_COST = 2
RENT_PRICE = 10
MAX_CARS = 20
GAMMA = 0.9
PMF_MATRIX_MAP = {expected: [poisson.pmf(occurred, expected) for occurred in range(11)] for expected in [2, 3, 4]}


class State:
    def __init__(self, nLoc1, nLoc2, terminated=False):
        self.nLoc1 = nLoc1
        self.nLoc2 = nLoc2
        self.terminated = terminated


def step(action, updates, state):
    """
    :param action: negative values for moving to location 1
    :param updates tuple (req1, req2, ret1, ret2)
    :param state: current state, not modidified
    :return: tuple of new state and reward
    """

    rented = 0
    nLoc1 = state.nLoc1
    nLoc2 = state.nLoc2

    req1, req2, ret1, ret2 = updates

    preq1 = PMF_MATRIX_MAP.get(LOC1_EXPECTED_REQUEST)[req1]
    pret1 = PMF_MATRIX_MAP.get(LOC1_EXPECTED_RETURN)[ret1]

    preq2 = PMF_MATRIX_MAP.get(LOC2_EXPECTED_REQUEST)[req2]
    pret2 = PMF_MATRIX_MAP.get(LOC2_EXPECTED_RETURN)[ret2]

    prob = preq1 * preq2 * pret1 * pret2

    rented += min(nLoc1, req1)
    rented += min(nLoc2, req2)

    nLoc1 -= req1
    nLoc2 -= req2

    if nLoc1 < 0 or nLoc2 < 0:
        return (State(-1, -1, True), 0, prob)

    nLoc1 = int(min(nLoc1 + ret1 - action, MAX_CARS))
    nLoc2 = int(min(nLoc2 + ret2 + action, MAX_CARS))

    return (
        State(nLoc1, nLoc2, False), rented * RENT_PRICE - abs(action) * CAR_MOVE_COST, prob)


def states():
    for nLoc1 in range(MAX_CARS + 1):
        for nLoc2 in range(MAX_CARS + 1):
            yield State(nLoc1, nLoc2)


def states_updates():
    for req1 in range(1, 6):
        for req2 in range(1, 4):
            for ret1 in range(1, 6):
                for ret2 in range(2, 7):
                    yield (req1, req2, ret1, ret2)


def policy_evaluation(V, pi):
    print("policy_evaluation started")

    teta = 0.0001
    delta = 1
    iteration = 0
    # policy evaluation
    while delta > teta:
        delta = 0
        iteration += 1

        for s in states():
            v = V[s.nLoc1, s.nLoc2]
            v_new = 0
            v_new_updates = []
            for u in states_updates():
                sp, r, p = step(pi[s.nLoc1, s.nLoc2], u, s)
                vsp = 0 if sp.terminated else V[sp.nLoc1, sp.nLoc2]
                # v_new_updates.append((r, p, vsp, p * (r + GAMMA * vsp)))
                v_new += p * (r + GAMMA * vsp)

            V[s.nLoc1, s.nLoc2] = v_new

            delta = max(abs(V[s.nLoc1, s.nLoc2] - v), delta)

        if iteration % 10 == 0:
            print("policy_evaluation iteration {}, max delta='{}'".format(iteration, delta))


def policy_improvement(V, pi):
    print("policy_improvement started")

    policy_stable = True

    for s in states():
        old_action = pi[s.nLoc1, s.nLoc2]
        actions = np.arange(-s.nLoc2, s.nLoc1 + 1)  # negative actions goes to location 1
        action_values = np.empty_like(actions)

        for a in range(len(actions)):
            sum = 0
            for u in states_updates():
                sp, r, p = step(actions[a], u, s)
                vsp = 0 if sp.terminated else V[sp.nLoc1, sp.nLoc2]
                sum += p * (r + GAMMA * vsp)
            action_values[a] = sum

        pi[s.nLoc1, s.nLoc2] = actions[np.argmax(action_values)]

        if pi[s.nLoc1, s.nLoc2] != old_action: policy_stable = False

    return policy_stable


def policy_iteration():
    V = numpy.zeros((MAX_CARS + 1, MAX_CARS + 1))
    pi = numpy.zeros((MAX_CARS + 1, MAX_CARS + 1))

    policy_stable = False

    while not policy_stable:
        policy_evaluation(V, pi)
        policy_stable = policy_improvement(V, pi)

    np.savetxt("v.txt", V)
    np.savetxt("pi.txt", pi)


if __name__ == '__main__':
    policy_iteration()
