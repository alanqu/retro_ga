import retro  # register the universe environments
import numpy as np
import random
import argparse
import copy


def rollout(env, acts):
	"""
	Perform a rollout using a preset collection of actions
	"""
	total_rew = 0
	env.reset()
	steps = 0
	for act in acts:
		_obs, rew, done, _info = env.step(act)
		env.render()
		steps += 1
		total_rew += rew
		if done:
			break

	return steps, total_rew


class Action:
	def __init__(self, acts, rew):
		self.acts = acts
		self.rew = rew


class GA:

	def __init__(self, game, max_episode_steps, max_total_steps, 
				state, scenario):
		self.env = retro.make(game, state, use_restricted_actions=retro.Actions.DISCRETE,
			scenario=scenario)
		self.env.reset()
		self.best_rew = float('-inf')
		self.max_episode_steps = max_episode_steps
		self.max_total_steps = max_total_steps
		self.acts_pool = []
		# The pool needs 2 actions to begin
		for i in range(0, 2):
			acts = []
			for i in range(self.max_episode_steps):
				acts.append(self.env.action_space.sample())
			self.acts_pool.append(Action(acts, 0))

	def selection(self):
		rand = -1
		index = -1
		while (rand < 0) or (index < 0) or (index > len(self.acts_pool) - 1):
			rand = random.normalvariate(0, len(self.acts_pool)/float(6))
			index = round(rand)
		print("selected {} from the list".format(str(index)))
		return self.acts_pool[index].acts

	def crossover(self, p_1_acts, p_2_acts):
		result_acts = []
		for i in range(min(len(p_1_acts), len(p_2_acts)) - 1):
			rand = random.randint(0, 1)
			if rand == 0:
				result_acts.append(p_1_acts[i])
			else:
				result_acts.append(p_2_acts[i])
		return result_acts

	def mutation(self, acts):
		result_acts = []
		for i in range(len(acts) - 1):
			rand = random.randint(0, 65) # 1.5%
			if rand == 0:
				# mutate to random key
				result_acts.append(self.env.action_space.sample())
			else:
				result_acts.append(acts[i])
		return result_acts 

	def insert_to_pool(self, acts, rew):
		action_obj = Action(acts, rew)
		if len(self.acts_pool) <= 0:
			self.acts_pool.append(Action(acts, rew))
			return

		for item in self.acts_pool:
			if item.rew < rew:
				fh = self.acts_pool[:self.acts_pool.index(item) + 1]
				lh = self.acts_pool[self.acts_pool.index(item) + 1:]
				self.acts_pool = copy.deepcopy(fh)
				self.acts_pool.append(action_obj)
				if lh is not None:
					self.acts_pool.extend(lh)
				if len(self.acts_pool) > 100:
					self.acts_pool.pop()
				return

		self.acts_pool.append(Action(acts, rew))
		return

	def select_actions(self):
		result_acts = []
		p_1_acts = self.selection()
		p_2_acts = self.selection()
		result_acts = self.crossover(p_1_acts, p_2_acts)
		result_acts = self.mutation(result_acts)
		return result_acts

	def run(self):
		count = 0
		best_rec = "best_{}.bk2"
		timesteps = 0
		while True:
			acts = self.select_actions()
			steps, rew = rollout(self.env, acts)
			self.insert_to_pool(acts, rew)
			timesteps += len(acts)

			if rew > self.best_rew:
				print("new best reward {} => {}".format(self.best_rew, rew))
				self.best_rew = rew
				self.env.unwrapped.record_movie(best_rec.format(str(count)))
				self.env.reset()
				for act in acts:
					self.env.step(act)
				self.env.unwrapped.stop_record()

			count += 1
			if timesteps > self.max_total_steps:
				print("max steps exceeded")
				break


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--game', default='SuperMarioBros-Nes')
	parser.add_argument('--state', default=retro.State.DEFAULT)
	parser.add_argument('--scenario', default=None)
	parser.add_argument('--max_episode_steps', default=4500)
	parser.add_argument('--max_total_steps', default=1e8)
	args = parser.parse_args()

	ga = GA(game=args.game, max_episode_steps=args.max_episode_steps,
		max_total_steps=int(args.max_total_steps), state=args.state, scenario=args.scenario)
	ga.run()


if __name__ == "__main__":
	main()