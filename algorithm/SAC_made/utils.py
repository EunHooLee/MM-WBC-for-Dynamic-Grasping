env = NormalizedActions(gym.make(args.env_name))

# Set seeds
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_Val = torch.tensor(1e-7).float()
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'd'])