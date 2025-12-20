import csv
import time
import numpy as np
from pathlib import Path
import torch
from itertools import chain
from runners.separated.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class CRunner(Runner):
    """Runner class to perform training, evaluation. See parent class for details."""
    def __init__(self, config):
        super(CRunner, self).__init__(config)
        np.random.seed(42)
        self.completed_episodes = 0
        self.env_infos_buffer = {}
        # Track latest per-episode metrics (updated by _record_episode_metrics)
        self._latest_metrics = {}

        # CSV path & header (include reward explicitly)
        self.metrics_path = self.run_dir / "env_metrics.csv"
        self.metrics_header = ["episode", "reward", "service_level"]
        for agent in range(self.num_agents):
            for rule_idx in range(3):
                self.metrics_header.append(f"agent{agent}_rule{rule_idx}")
        for agent in range(self.num_agents):
            self.metrics_header.append(f"agent{agent}_order_variance")
        for agent in range(self.num_agents):
            self.metrics_header.append(f"agent{agent}_bullwhip")
        if not self.metrics_path.exists():
            with open(self.metrics_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(self.metrics_header)

    def run(self):

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        rewards_log = []
        inv_log = []
        actions_log = []
        demand_log = []
        overall_reward= []
        best_reward = float('-inf')
        best_bw = []
        record = 0
        episode_rewards = []

        for episode in range(episodes):
            
            if episode % self.eval_interval == 0 and self.use_eval:
                re, bw_res = self.eval()
                print()
                print("Eval average reward: ", re, " Eval ordering fluctuation measurement (downstream to upstream): ", bw_res)
                if(re > best_reward and episode > 0):
                    self.save()
                    print("A better model is saved!")
                    best_reward = re
                    best_bw = bw_res
                    record = 0
                elif(episode > self.n_warmup_evaluations):
                    record += 1
                    if(record == self.n_no_improvement_thres):
                        print("Training finished because of no imporvement for " + str(self.n_no_improvement_thres) + " evaluations")
                        return best_reward, best_bw

            self.warmup()
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            # reset per-episode trackers
            self._latest_metrics = {}
            episode_reward_sum = 0.0

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                self._update_env_metrics(infos)
                
                share_obs = []
                for o in obs:
                    share_obs.append(list(chain(*o)))
                
                available_actions = np.array([[None for agent_id in range(self.num_agents)] for info in infos])

                rewards_log.append(rewards)

                episode_reward_sum += np.mean(rewards)

                inv, demand, orders = self.envs.get_property()
                inv_log.append(inv)
                demand_log.append(demand)
                actions_log.append(orders)

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # finalize episode metrics
            avg_episode_reward = episode_reward_sum / float(self.episode_length)
            episode_rewards.append(avg_episode_reward)

            # build per-episode row
            row = [episode, avg_episode_reward]
            service = float(self._latest_metrics.get("service_level", 0.0))
            service = min(1.0, max(0.0, service))
            row.append(service)

            for agent in range(self.num_agents):
                usage = self._latest_metrics.get("rule_usage", [])
                usage_agent = usage[agent] if agent < len(usage) else []
                for rule_idx in range(3):
                    value = float(usage_agent[rule_idx]) if rule_idx < len(usage_agent) else 0.0
                    row.append(value)

            for agent in range(self.num_agents):
                order_var = self._latest_metrics.get("order_variance", [])
                value = float(order_var[agent]) if agent < len(order_var) else 0.0
                row.append(value)

            for agent in range(self.num_agents):
                bullwhip = self._latest_metrics.get("bullwhip_index", [])
                value = float(bullwhip[agent]) if agent < len(bullwhip) else 0.0
                row.append(value)

            with open(self.metrics_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                threads_rew = [[] for i in range(self.n_rollout_threads)]
                threads_inv = [[] for i in range(self.n_rollout_threads)]
                threads_act = [[] for i in range(self.n_rollout_threads)]
                threads_demand = [[] for i in range(self.n_rollout_threads)]
                for i in range(len(rewards_log)):
                    for j in range(self.n_rollout_threads):
                        threads_rew[j].append(rewards_log[i][j])
                        threads_inv[j].append(inv_log[i][j])
                        threads_act[j].append(actions_log[i][j])
                        threads_demand[j].append(demand_log[i][j])
                
                overall_reward.append(np.mean(threads_rew))
                if(len(overall_reward)<6):
                    smooth_reward = overall_reward
                else:
                    smooth_reward = []
                    for i in range(len(overall_reward)-5):
                        smooth_reward.append(np.mean(overall_reward[i:i+10]))
                
                for t in range(len(threads_rew)):
                    rew = [[] for i in range(self.num_agents)]
                    inv = [[] for i in range(self.num_agents)]
                    act = [[] for i in range(self.num_agents)]
                    for i in range(len(threads_rew[t])):
                        for j in range(self.num_agents):
                            rew[j].append(threads_rew[t][i][j])
                            inv[j].append(threads_inv[t][i][j])
                            act[j].append(threads_act[t][i][j])
                    rew = [round(np.mean(l), 2) for l in rew]
                    inv = [round(np.mean(l), 2) for l in inv]
                    act = [round(np.mean(l), 2) for l in act]
                    print("Reward for thread " + str(t+1) + ": " + str(rew) + " " + str(round(np.mean(rew),2))+"  Inventory: " + str(inv)+"  Order: " + str(act) + " Demand: " + str(np.mean(threads_demand[t], 0)))
                self.log_env(self.env_infos_buffer, total_num_steps)
                self.env_infos_buffer = {}
                rewards_log = []
                inv_log = []
                actions_log = []
                demand_log = []

        return (float(np.mean(overall_reward)) if overall_reward else 0.0), best_bw
        # eval

    def _update_env_metrics(self, infos):
        for env_info in infos:
            if not env_info:
                continue
            first_info = env_info[0] if isinstance(env_info, (list, tuple)) else env_info
            if not isinstance(first_info, dict):
                continue
            metrics = first_info.get("episode_metrics")
            if metrics:
                self._record_episode_metrics(metrics)

    def _record_episode_metrics(self, metrics):
        service = float(metrics.get("service_level", 0.0))
        self._latest_metrics["service_level"] = service
        self.env_infos_buffer.setdefault("service_level", []).append(service)

        rule_usage = metrics.get("rule_usage", [])
        self._latest_metrics["rule_usage"] = rule_usage
        for agent in range(self.num_agents):
            usage = rule_usage[agent] if agent < len(rule_usage) else []
            for rule_idx in range(3):
                value = float(usage[rule_idx]) if rule_idx < len(usage) else 0.0
                key = f"rule_usage/agent_{agent}_rule_{rule_idx}"
                self.env_infos_buffer.setdefault(key, []).append(value)

        order_variance = metrics.get("order_variance", [])
        self._latest_metrics["order_variance"] = order_variance
        for agent in range(self.num_agents):
            value = float(order_variance[agent]) if agent < len(order_variance) else 0.0
            key = f"order_variance/agent_{agent}"
            self.env_infos_buffer.setdefault(key, []).append(value)

        bullwhip = metrics.get("bullwhip_index", [])
        self._latest_metrics["bullwhip_index"] = bullwhip
        for agent in range(self.num_agents):
            value = float(bullwhip[agent]) if agent < len(bullwhip) else 0.0
            key = f"bullwhip_index/agent_{agent}"
            self.env_infos_buffer.setdefault(key, []).append(value)
        # completed_episodes used only for logging; not incremented here to avoid mismatch

    def warmup(self):
        # reset env
        obs, available_actions = self.envs.reset()
        # replay buffer
        
        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents): 
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()
            self.buffer[agent_id].available_actions[0] = None

    @torch.no_grad()
    def collect(self, step):
        value_collector=[]
        action_collector=[]
        temp_actions_env = []
        action_log_prob_collector=[]
        rnn_state_collector=[]
        rnn_state_critic_collector=[]

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                self.buffer[agent_id].obs[step],
                                                self.buffer[agent_id].rnn_states[step],
                                                self.buffer[agent_id].rnn_states_critic[step],
                                                self.buffer[agent_id].masks[step],
                                                self.buffer[agent_id].available_actions[step])

            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))

            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                action_indices = action.cpu().detach().numpy().astype(int).reshape(-1)
                action_env = np.eye(self.envs.action_space[agent_id].n)[action_indices]
            else:
                raise NotImplementedError
            
            temp_actions_env.append(action_env)

            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))
        
        # [self.envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
                
            self.buffer[agent_id].insert(share_obs, np.array(list(obs[:, agent_id])), rnn_states[:,agent_id],
                    rnn_states_critic[:,agent_id],actions[:,agent_id], action_log_probs[:,agent_id],
                    values[:,agent_id], rewards[:,agent_id], masks[:,agent_id])

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            train_infos[agent_id]["average_step_rewards"] = np.mean(self.buffer[agent_id].rewards)
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self):
        
        overall_reward = []
        eval_num = self.eval_envs.get_eval_num()

        for _ in range(eval_num):
            eval_obs, eval_available_actions = self.eval_envs.reset()
            
            eval_share_obs = []
            for o in eval_obs:
                eval_share_obs.append(list(chain(*o)))
            eval_share_obs = np.array(eval_share_obs)

            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for eval_step in range(self.episode_length):
                temp_actions_env = []

                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].prep_rollout()
                    eval_actions, temp_rnn_state = \
                        self.trainer[agent_id].policy.act(eval_obs[:,agent_id],
                                                eval_rnn_states[:,agent_id],
                                                eval_masks[:,agent_id],
                                                None,
                                                deterministic=True)
                    eval_rnn_states[:,agent_id]=_t2n(temp_rnn_state)
                    action = eval_actions.detach().cpu().numpy()

                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)

                #eval_actions = np.array(eval_actions_collector).transpose(1,0,2)
                eval_actions_env = []
                for i in range(self.n_eval_rollout_threads):
                    eval_one_hot_action_env = []
                    for eval_temp_action_env in temp_actions_env:
                        eval_one_hot_action_env.append(eval_temp_action_env[i])
                    eval_actions_env.append(eval_one_hot_action_env)

                # Obser reward and next obs
                eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
                
                eval_share_obs = []
                for o in eval_obs:
                    eval_share_obs.append(list(chain(*o)))
                eval_share_obs = np.array(eval_share_obs)

                eval_available_actions = None

                overall_reward.append(np.mean(eval_rewards))

                eval_dones_env = np.all(eval_dones, axis=1)

                eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

                eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        
        bw_res = self.eval_envs.get_eval_bw_res()
        return np.mean(overall_reward), bw_res
