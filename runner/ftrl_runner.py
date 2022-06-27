import numpy as np


def run_ftrl(p_id, game, agents, n_iterations, logger, feedback='full', random_policy=False):
    trajectories = [[] for _ in agents]
    time_average_trajectories = [[] for _ in agents]
    exploitabilities = []
    time_average_exploitabilities = []
    index = []
    if random_policy:
        for i_a, agent in enumerate(agents):
            agent.policy = random_init_policy(game.payoff.shape[i_a])
    for i_t in np.arange(0, n_iterations + 1):
        if feedback == 'full':
            policies = [agent.policy for agent in agents]
            utilities = game.full_feedback(policies)
            for i_a, agent in enumerate(agents):
                agent.update(utilities[i_a])
        elif feedback == 'bandit':
            policies = [agent.policy for agent in agents]
            utilities, actions = game.bandit_feedback(policies)
            for i_a, agent in enumerate(agents):
                agent.update_bandit(utilities[i_a], actions[i_a])
        else:
            raise RuntimeError('illegal feedback type')
        index.append(i_t)
        for i_a, agent in enumerate(agents):
            trajectories[i_a].append(agent.policy.copy())
            time_average_trajectories[i_a].append(agent.time_average_policy.copy())
            time_average_policies = [agent.time_average_policy for agent in agents]
        exploitabilities.append(game.calc_exploitability(policies))
        time_average_exploitabilities.append(game.calc_exploitability(time_average_policies))
        if i_t > 0 and i_t % int(10e5) == 0:
            logger.write_trajectories(trajectories, index)
            logger.write_time_avarage_trajectoies(time_average_trajectories, index)
            logger.write_exploitabilities(exploitabilities, index)
            logger.write_time_average_exploitabilities(time_average_exploitabilities, index)
            if p_id % 10 == 0:
                print('p_id', p_id, ":", i_t, "iterations finished.")
    logger.write_trajectories(trajectories, index)
    logger.write_time_avarage_trajectoies(time_average_trajectories, index)
    logger.write_exploitabilities(exploitabilities, index)
    logger.write_time_average_exploitabilities(time_average_exploitabilities, index)

    return agents


def random_init_policy(n_actions):
    random_numbers = np.random.exponential(scale=1.0, size=n_actions)
    return np.array(random_numbers / random_numbers.sum(), dtype=np.float64)
