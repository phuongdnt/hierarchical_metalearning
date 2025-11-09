from envs.serial_rulechoice import Env

def test_reset_observation_shape():
    env = Env()
    obs = env.reset()
    assert len(obs) == env.agent_num
    for agent_obs in obs:
        assert agent_obs.shape == (env.obs_dim,)

def test_action_map_uses_rule_parameters():
    env = Env()
    env.reset()
    env.inventory = [0.0 for _ in range(env.agent_num)]
    env.backlog = [0.0 for _ in range(env.agent_num)]
    env.pipeline = [[0.0 for _ in env.pipeline[0]] for _ in range(env.agent_num)]
    env.demand_history = [10.0, 10.0, 10.0]
    orders = env.action_map([0, 1, 2])
    assert orders[0] > 0
    assert orders[1] >= 0
    assert orders[2] >= 0
    assert env.rule_counts[0][0] == 1
    assert env.rule_counts[1][1] == 1

def test_step_updates_time_since_last_order():
    env = Env()
    env.reset()
    env.demand_list = [0.0 for _ in range(env.episode_max_steps)]
    env.inventory = [100.0 for _ in range(env.agent_num)]
    env.pipeline = [[0.0 for _ in env.pipeline[0]] for _ in range(env.agent_num)]
    obs, reward, done, infos = env.step([0, 0, 0], one_hot=False)
    assert env.time_since_last_order[0] == 1
    assert infos[0]['chosen_rule'] == 0
    assert infos[0]['order_quantity'] == env.last_order_qty[0]

def test_episode_metrics_emitted_on_done():
    env = Env()
    env.reset()
    env.episode_max_steps = 1
    env.demand_list = [5.0]
    env.pipeline = [[0.0 for _ in env.pipeline[0]] for _ in range(env.agent_num)]
    obs, reward, done, infos = env.step([0, 0, 0], one_hot=False)
    assert done[0] is True
    metrics = infos[0].get('episode_metrics')
    assert metrics is not None
    assert 'rule_usage' in metrics
    assert 'service_level' in metrics
