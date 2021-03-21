from copy import copy
from collections import deque
import utils
import numpy as np
from pddlgym.core import get_successor_states, InvalidAction
from pddlgym.inference import check_goal


def add_state_graph(s, graph, to_str=False, add_expanded_prop=False):
    graph_ = copy(graph)
    graph_[str(s) if to_str else s] = {'Adj': []}

    return graph_


def get_successor_states_check_exception(s, a, domain, return_probs=True):
    try:
        succ = get_successor_states(s,
                                    a,
                                    domain,
                                    raise_error_on_invalid_action=True,
                                    return_probs=return_probs)
    except InvalidAction:
        succ = {s: 1.0} if return_probs else frozenset({s})

    return succ


def get_all_reachable(s, A, env, reach=None):
    reach = {} if not reach else reach

    reach[s] = {}
    for a in A:
        succ = get_successor_states_check_exception(s, a, env.domain)

        reach[s][a] = {s_: prob for s_, prob in succ.items()}
        for s_ in succ:
            if s_ not in reach:
                reach.update(get_all_reachable(s_, A, env, reach))
    return reach

def expand_state(s, h_v, env, explicit_graph, goal, A, succs_cache=None):
    if check_goal(s, goal):
        raise ValueError(
            f'State {s} can\'t be expanded because it is a goal state')

    neighbour_states_dict = {}
    neighbour_states = []
    i = 0
    for a in A:
        if succs_cache and (s, a) in succs_cache:
            succs = succs_cache[(s, a)]
        else:
            succs = get_successor_states_check_exception(s, a, env.domain)
        for s_, p in succs.items():
            if s_ not in neighbour_states_dict:
                neighbour_states_dict[s_] = i
                i += 1
                neighbour_states.append({'state': s_, 'A': {a: p}})
            else:
                neighbour_states[neighbour_states_dict[s_]]['A'][a] = p

    unexpanded_neighbours = filter(
        lambda _s: (not _s['state'] in explicit_graph) or
        (not explicit_graph[_s['state']]['expanded']), neighbour_states)

    # Add new empty states to 's' adjacency list
    new_explicit_graph = copy(explicit_graph)

    new_explicit_graph[s]["Adj"].extend(neighbour_states)

    for n in unexpanded_neighbours:
        if n['state'] != s:
            is_goal = check_goal(n['state'], goal)
            h_v_ = 0 if is_goal else h_v(n['state'])
            new_explicit_graph[n['state']] = {
                "value": h_v_,
                "solved": False,
                "pi": None,
                "expanded": False,
                "Q_v": {a: h_v_
                        for a in A},
                "Adj": []
            }

    new_explicit_graph[s]['expanded'] = True

    return new_explicit_graph


def get_unexpanded_states(goal, explicit_graph, bpsg):
    return list(
        filter(
            lambda x: (x not in explicit_graph) or
            (not explicit_graph[x]["expanded"] and not check_goal(x, goal)),
            bpsg.keys()))


def find_reachable(s, a, mdp):
    """ Find states that are reachable from state 's' after executing action 'a' """
    all_reachable_from_s = mdp[s]['Adj']
    return list(filter(lambda obj_s_: a in obj_s_['A'], all_reachable_from_s))


def find_direct_ancestors(s, graph, visited, best=False):
    return list(
        filter(
            lambda s_: s_ != s and (s_ not in visited) and len(
                list(
                    filter(
                        lambda s__: s__['state'] == s and
                        (True if not best else graph[s_]['pi'] in s__['A']
                         ), graph[s_]['Adj']))) > 0, graph))


def __find_ancestors(s, bpsg, visited, best):
    # Find states in graph that have 's' in the adjacent list (except from 's' itself and states that were already visited):
    direct_ancestors = list(
        filter(lambda a: a not in visited,
               find_direct_ancestors(s, bpsg, visited, best)))

    result = [] + direct_ancestors

    for a in direct_ancestors:
        if a not in visited:
            visited.add(a)
            result += __find_ancestors(a, bpsg, visited, best)

    return result


def find_ancestors(s, bpsg, best=False):
    return __find_ancestors(s, bpsg, set(), best)


def find_neighbours(s, adjs):
    """ Find neighbours of s in adjacency list (except itself) """
    return list(
        map(lambda s_: s_['state'], filter(lambda s_: s_['state'] != s, adjs)))


def find_unreachable(s0, mdp):
    """ search for unreachable states using dfs """
    S = list(mdp.keys())
    len_s = len(S)
    V_i = {S[i]: i for i in range(len_s)}
    colors = ['w'] * len_s
    dfs_visit(V_i[s0], colors, [-1] * len_s, [-1] * len_s, [-1] * len_s, [0],
              S, V_i, mdp)
    return [S[i] for i, c in enumerate(colors) if c != 'b']


def dfs_visit(i,
              colors,
              d,
              f,
              low,
              time,
              S,
              V_i,
              mdp,
              on_visit=None,
              on_visit_neighbor=None,
              on_finish=None):
    colors[i] = 'g'
    time[0] += 1
    d[i] = time[0]
    low[i] = time[0]
    s = S[i]

    if on_visit:
        on_visit(s, i, d, low)

    for s_obj in mdp[s]['Adj']:
        s_ = s_obj['state']
        if s_ not in mdp:
            continue
        j = V_i[s_]
        if colors[j] == 'w':
            dfs_visit(j, colors, d, f, low, time, S, V_i, mdp, on_visit,
                      on_visit_neighbor, on_finish)
            low[i] = min(low[i], low[j])
        if on_visit_neighbor:
            on_visit_neighbor(s, i, s_, j, d, low)

    if on_finish:
        on_finish(s, i, d, low)

    colors[i] = 'b'
    time[0] += 1
    f[i] = time[0]


def dfs(mdp, on_visit=None, on_visit_neighbor=None, on_finish=None):
    S = list(mdp.keys())
    len_s = len(S)
    V_i = {S[i]: i for i in range(len_s)}
    # (w)hite, (g)ray or (b)lack
    colors = ['w'] * len_s
    d = [-1] * len_s
    f = [-1] * len_s
    low = [-1] * len_s
    time = [0]
    for i in range(len_s):
        c = colors[i]
        if c == 'w':
            dfs_visit(i, colors, d, f, low, time, S, V_i, mdp, on_visit,
                      on_visit_neighbor, on_finish)

    return d, f, colors

def update_action_partial_solution(s, s0, bpsg, explicit_graph):
    """
        Updates partial solution given pair of state and action
    """
    bpsg_ = copy(bpsg)
    i = 0
    states = [s]
    while len(states) > 0:
        s = states.pop()
        a = explicit_graph[s]['pi']
        s_obj = bpsg_[s]

        s_obj['Adj'] = []
        reachable = find_reachable(s, a, explicit_graph)

        for s_obj_ in reachable:
            s_ = s_obj_['state']
            s_obj['Adj'].append({'state': s_, 'A': {a: s_obj_['A'][a]}})
            if s_ not in bpsg_:
                bpsg_ = add_state_graph(s_, bpsg_)
                bpsg_[s] = s_obj

                if explicit_graph[s_]['expanded']:
                    states.append(s_)
        i += 1

    return bpsg_


def update_partial_solution(s0, bpsg, explicit_graph):
    bpsg_ = copy(bpsg)

    for s in bpsg:
        a = explicit_graph[s]['pi']
        if s not in bpsg_:
            continue

        s_obj = bpsg_[s]

        if len(s_obj['Adj']) == 0:
            if a is not None:
                bpsg_ = update_action_partial_solution(s, s0, bpsg_,
                                                       explicit_graph)
        else:
            best_current_action = next(iter(s_obj['Adj'][0]['A'].keys()))

            if a is not None and best_current_action != a:
                bpsg_ = update_action_partial_solution(s, s0, bpsg_,
                                                       explicit_graph)

    unreachable = find_unreachable(s0, bpsg_)

    for s_ in unreachable:
        if s_ in bpsg_:
            bpsg_.pop(s_)

    return bpsg_

def backup_bellman(explicit_graph, A, s, goal, gamma, C):
    np.seterr(all='raise')

    all_reachable = np.array([find_reachable(s, a, explicit_graph) for a in A],
                             dtype=object)

    actions_results = np.array([
        np.sum([
            C(s, A[i]) + gamma * explicit_graph[s_['state']]['value'] * s_['A'][a] for s_ in all_reachable[i]
        ]) for i, a in enumerate(A)
    ])


    i_a = np.argmin(actions_results)
    explicit_graph[s]['value'] = actions_results[i_a]
    explicit_graph[s]['pi'] = A[i_a]

    return explicit_graph


def value_iteration(explicit_graph,
                    bpsg,
                    A,
                    Z,
                    goal,
                    gamma,
                    C,
                    epsilon=1e-3,
                    p_zero=True,
                    n_iter=None,
                    convergence_test=False):
    n_states = len(explicit_graph)
    n_actions = len(A)

    # initialize
    V = np.zeros(n_states, dtype=float)
    pi = np.full(n_states, None)
    V_i = {s: i for i, s in enumerate(explicit_graph)}
    A_i = {a: i for i, a in enumerate(A)}

    for s, n in explicit_graph.items():
        V[V_i[s]] = n['value']
        pi[V_i[s]] = n['pi']

    i = 0

    V_ = np.copy(V)
    pi_ = np.copy(pi)

    changed = False
    converged = False
    n_updates = 0
    np.seterr(all='raise')
    while True:
        for s in Z:
            if explicit_graph[s]['solved']:
                continue

            n_updates += 1
            all_reachable = np.array(
                [find_reachable(s, a, explicit_graph) for a in A],
                dtype=object)

            actions_results = np.array([
                np.sum([
                    C(s, A[i]) + gamma * V[V_i[s_['state']]] * s_['A'][a]
                    for s_ in all_reachable[i]
                ]) for i, a in enumerate(A)
            ])

            i_a = np.argmin(actions_results)
            V_[V_i[s]] = actions_results[i_a]
            pi_[V_i[s]] = A[i_a]

        v_norm = np.linalg.norm(V_[list(V_i.values())] - V[list(V_i.values())],
                                np.inf)

        different_actions = pi_[list(
            V_i.values())][pi_[list(V_i.values())] != pi[list(V_i.values())]]
        if len(different_actions) > 0:
            changed = True
        if v_norm < epsilon:
            converged = True
        V = np.copy(V_)
        pi = np.copy(pi_)

        if converged:
            print(
                f'{convergence_test}, {changed}, {converged}, {v_norm}'
            )
            break

        i += 1

    # save results in explicit graph
    for s in Z:
        explicit_graph[s]['value'] = V[V_i[s]]
        explicit_graph[s]['pi'] = pi[V_i[s]]

    #print(f'{i} iterations')
    return explicit_graph, converged, changed, n_updates

def lao(s0, h_v, goal, A, gamma, env, epsilon=1e-3):
    bpsg = {s0: {"Adj": []}}
    explicit_graph = {}

    explicit_graph[s0] = {
        "value": h_v(s0),
        "solved": False,
        "expanded": False,
        "pi": None,
        "Q_v": {a: h_v(s0)
                for a in A},
        "Adj": []
    }

    def C(s, a):
        return 0 if check_goal(s, goal) else 1

    i = 1

    unexpanded = get_unexpanded_states(goal, explicit_graph, bpsg)
    n_updates = 0
    explicit_graph_cur_size = 1
    while True:
        while len(unexpanded) > 0:
            s = unexpanded[0]
            print("Iteration", i)
            print("Will expand", len(unexpanded), "states")
            Z = set()
            for s in unexpanded:
                explicit_graph = expand_state(s, h_v, env, explicit_graph,
                                              goal, A)
                Z.add(s)
                Z.update(find_ancestors(s, explicit_graph, best=True))

            assert len(explicit_graph) >= explicit_graph_cur_size
            explicit_graph_cur_size = len(explicit_graph)
            print("explicit graph size:", explicit_graph_cur_size)
            print("Z size:", len(Z))
            explicit_graph, _, __, n_updates_ = value_iteration(
                explicit_graph,
                bpsg,
                A,
                Z,
                goal,
                gamma,
                C,
                epsilon=epsilon)
            print(f"Finished value iteration in {n_updates_} updates")
            n_updates += n_updates_
            bpsg = update_partial_solution(s0, bpsg, explicit_graph)
            unexpanded = get_unexpanded_states(goal, explicit_graph, bpsg)
            i += 1
        bpsg_states = [s_ for s_ in bpsg.keys() if not check_goal(s_, goal)]
        print(f"Will start convergence test for bpsg with {len(bpsg)} states")
        explicit_graph, converged, changed, n_updates_ = value_iteration(
            explicit_graph,
            bpsg,
            A,
            bpsg_states,
            goal,
            gamma,
            C,
            epsilon=epsilon,
            convergence_test=True)
        print(f"Finished convergence test in {n_updates_} updates")
        n_updates += n_updates_

        bpsg = update_partial_solution(s0, bpsg, explicit_graph)
        unexpanded = get_unexpanded_states(goal, explicit_graph, bpsg)

        if converged and len(unexpanded) == 0:
            break
    return explicit_graph, bpsg, n_updates


def ilao(s0,
         h_v,
         goal,
         A,
         gamma,
         env,
         epsilon=1e-3,
         succs_cache=None):

    bpsg = {s0: {"Adj": []}}
    explicit_graph = {}
    succs_cache = {} if succs_cache == None else succs_cache

    explicit_graph[s0] = {
        "value": h_v(s0),
        "solved": False,
        "expanded": False,
        "pi": None,
        "Q_v": {a: h_v(s0)
                for a in A},
        "Adj": []
    }

    def C(s, a):
        return 0 if check_goal(s, goal) else 1

    i = 1
    unexpanded = get_unexpanded_states(goal, explicit_graph, bpsg)
    n_updates = 0
    explicit_graph_cur_size = 1
    while True:
        while len(unexpanded) > 0:
            print("Iteration", i)
            print(len(unexpanded), "unexpanded states")

            n_updates_ = 0

            def visit(s, i, d, low):
                nonlocal explicit_graph, A, goal, n_updates_
                is_goal = check_goal(s, goal)
                if not is_goal and not explicit_graph[s]['expanded']:
                    explicit_graph = expand_state(
                        s,
                        h_v,
                        env,
                        explicit_graph,
                        goal,
                        A,
                        succs_cache=succs_cache)
                if not is_goal:
                    # run bellman backup
                    explicit_graph = backup_bellman(
                        explicit_graph, A, s, goal, gamma, C)
                    n_updates_ += 1

            dfs(bpsg, on_visit=visit)

            assert len(explicit_graph) >= explicit_graph_cur_size

            explicit_graph_cur_size = len(explicit_graph)
            print("explicit graph size:", explicit_graph_cur_size)
            print(f"Finished value iteration in {n_updates_} updates")
            n_updates += n_updates_
            bpsg = update_partial_solution(s0, bpsg, explicit_graph)

            unexpanded = get_unexpanded_states(goal, explicit_graph, bpsg)
            i += 1
        bpsg_states = [s_ for s_ in bpsg.keys() if not check_goal(s_, goal)]
        print(f"Will start convergence test for bpsg with {len(bpsg)} states")
        explicit_graph, converged, changed, n_updates_ = value_iteration(
            explicit_graph,
            bpsg,
            A,
            bpsg_states,
            goal,
            gamma,
            C,
            epsilon=epsilon,
            convergence_test=True)
        n_updates += n_updates_
        print(f"Finished convergence test in {n_updates_} updates")

        bpsg = update_partial_solution(s0, bpsg, explicit_graph)

        unexpanded = get_unexpanded_states(goal, explicit_graph, bpsg)

        if changed:
            continue

        if converged and len(unexpanded) == 0:
            break
    for s_ in bpsg:
        explicit_graph[s_]['solved'] = True
    return explicit_graph, bpsg, n_updates
