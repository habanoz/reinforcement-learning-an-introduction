import time
import numpy as np
from scipy.stats import poisson

LOC2_EXPECTED_RETURN = 4
LOC2_EXPECTED_REQUEST = 2
LOC1_EXPECTED_RETURN = 3
LOC1_EXPECTED_REQUEST = 3

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

    req1, req2, ret1, ret2 = updates.req1, updates.req2, updates.ret1, updates.ret2

    rented += min(nLoc1, req1)
    rented += min(nLoc2, req2)

    nLoc1 -= req1
    nLoc2 -= req2

    if nLoc1 < 0 or nLoc2 < 0:
        return (State(-1, -1, True), 0)

    nLoc1 = int(min(nLoc1 + ret1 - action, MAX_CARS))
    nLoc2 = int(min(nLoc2 + ret2 + action, MAX_CARS))

    return (
        State(nLoc1, nLoc2, False), rented * RENT_PRICE - abs(action) * CAR_MOVE_COST)


def states():
    for nLoc1 in range(MAX_CARS + 1):
        for nLoc2 in range(MAX_CARS + 1):
            yield State(nLoc1, nLoc2)


def states_updates(s):
    for req1 in range(0, s.nLoc1 + 1):
        for req2 in range(0, s.nLoc2 + 1):
            if USE_EXPECTED_RETURNS:
                yield ProbableUpdate(req1, PMF_MATRIX_MAP.get(LOC1_EXPECTED_REQUEST)[req1],
                                     req2, PMF_MATRIX_MAP.get(LOC2_EXPECTED_REQUEST)[req2],
                                     LOC1_EXPECTED_RETURN, 1.0,
                                     LOC2_EXPECTED_RETURN, 1.0
                                     )
            else:
                for ret1 in range(0, min(MAX_CARS - s.nLoc1, MAX_CARS_MOVED) + 1):
                    for ret2 in range(0, min(MAX_CARS - s.nLoc2, MAX_CARS_MOVED) + 1):
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
    # policy evaluation
    while delta > theta:
        delta = 0
        iteration += 1
        istart = time.time()

        for s in states():
            v = V[s.nLoc1, s.nLoc2]
            v_new = 0
            v_new_updates = []
            for u in states_updates(s):
                sp, r = step(pi[s.nLoc1, s.nLoc2], u, s)
                vsp = 0 if sp.terminated else V[sp.nLoc1, sp.nLoc2]
                # v_new_updates.append((r, p, vsp, p * (r + GAMMA * vsp)))
                prob = u.preq1 * u.preq2 * u.pret1 * u.pret2
                v_new += prob * (r + GAMMA * vsp)

            V[s.nLoc1, s.nLoc2] = v_new

            delta = max(abs(V[s.nLoc1, s.nLoc2] - v), delta)

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
        action_values = np.empty_like(actions)

        for a in range(len(actions)):
            sum = 0
            for u in states_updates(s):
                sp, r = step(actions[a], u, s)
                vsp = 0 if sp.terminated else V[sp.nLoc1, sp.nLoc2]

                prob = u.preq1 * u.preq2 * u.pret1 * u.pret2
                sum += prob * (r + GAMMA * vsp)
            action_values[a] = sum

        pi[s.nLoc1, s.nLoc2] = actions[np.argmax(action_values)]

        if pi[s.nLoc1, s.nLoc2] != old_action: policy_stable = False

    print("policy_improvement completed in {} seconds".format(time.time() - start))

    return policy_stable


def policy_iteration():
    V = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    pi = np.zeros((MAX_CARS + 1, MAX_CARS + 1))

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
