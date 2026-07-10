from .api import simulate_one_game, simulate_games, sweep_mana_bases

def demo_single(): return simulate_one_game(seed=1)
def demo_batch(): return simulate_games(games=10, seed=1)
def demo_sweep(): return sweep_mana_bases(games_per_config=10, limit_configs=5)
