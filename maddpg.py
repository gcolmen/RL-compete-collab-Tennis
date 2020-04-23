# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utils import soft_update, transpose_to_tensor, transpose_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

class MADDPG:
    def __init__(self, in_actor, out_actor, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()
 
        in_critic = in_actor + out_actor
        # critic input = obs_full + actions = 14+2+2+2=20
        self.maddpg_agent = [DDPGAgent(in_actor, out_actor, in_critic), 
                             DDPGAgent(in_actor, out_actor, in_critic)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        #actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        len_ = len(obs_all_agents)
        actions = [self.maddpg_agent[0].act(obs_all_agents[0:int(len_/2)], noise)]
        actions += [self.maddpg_agent[1].act(obs_all_agents[int(len_/2):], noise)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
#         target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        len_ = len(obs_all_agents)
        target_actions = [self.maddpg_agent[0].target_act(obs_all_agents[0:int(len_/2)], noise)]
        target_actions += [self.maddpg_agent[1].target_act(obs_all_agents[int(len_/2):], noise)]
        
        return target_actions

    def convert_to_tensor(self, input_list):
        list_ = [torch.tensor(x, dtype=torch.float) for x in input_list]
        #print(input_list)
        #list_ = torch.cat(list_, dim=1) # torch.cat(target_actions, dim=0)

        return list_

    def update(self, samples, agent_number):#, logger):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        #obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)
        states, next_states, action, reward, done = self.convert_to_tensor(samples)
        ##Stack states and next_states
        #obs_full = torch.stack(obs_full)
        #next_obs_full = torch.stack(next_obs_full)
#         print("nx:",next_states)
#         print("st:", states)
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        #target_actions = self.target_act(next_obs)
        target_actions = self.target_act(next_states)
        
        target_actions = torch.cat(target_actions, dim=0) # torch.cat(target_actions, dim=0)

#         print("tg:",target_actions)
#         print("nx:",next_states)
        
        #target_critic_input = torch.cat((next_obs_full.t(),target_actions), dim=1).to(device)
        target_critic_input = torch.cat((next_states, target_actions), dim=1).to(device)
        
#         print("tgcr:", target_critic_input)
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
#         action = torch.cat(action, dim=1)
#         action = torch.stack(action)

#         critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
        critic_input = torch.cat((states, action), dim=1).to(device)
        q = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
#         q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
#                    else self.maddpg_agent[i].actor(ob).detach()
#                    for i, ob in enumerate(obs) ]
        
        len_ = len(states)
        q_input = [ self.maddpg_agent[i].actor(states[0:int(len_/2)]) if i == agent_number \
                   else self.maddpg_agent[i].actor(states[int(len_/2):]).detach() \
                   for i in range(2) ]
                
#             len_ = len(obs_all_agents)
#         actions = [self.maddpg_agent[0].act(obs_all_agents[0:int(len_/2)], noise)]
#         actions += [self.maddpg_agent[1].act(obs_all_agents[int(len_/2):], noise)]
        
        
        q_input = torch.cat(q_input, dim=0)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
#         print("st",states)
#         print("inp",q_input)
        q_input2 = torch.cat((states, q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
#         logger.add_scalars('agent%i/losses' % agent_number,
#                            {'critic loss': cl,
#                             'actor_loss': al},
#                            self.iter)

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            