import gfootball.env as football_env

env = football_env.create_environment(env_name="academy_empty_goal_close", stacked=False, logdir='/tmp/football', write_goal_dumps=False, write_full_episode_dumps=False, render=False)
env.reset()
steps = 0
while True:
  action = env.action_space.sample()
  print(steps, action)
  obs, rew, done, info = env.step(action)
  steps += 1
  if steps % 10 == 0:
    print("Step %d Reward: %f" % (steps, rew))
  if done:
    break

print("Steps: %d Reward: %.2f" % (steps, rew))

traces = env.env.env.env.env._env._trace._trace
last_trace = traces[-1]
# https://github.com/google-research/football/blob/master/gfootball/doc/observation.md
last_observation = last_trace._trace['observation']
print(last_observation)

