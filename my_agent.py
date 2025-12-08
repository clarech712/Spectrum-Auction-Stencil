from agt_server.agents.base_agents.lsvm_agent import MyLSVMAgent
from agt_server.local_games.lsvm_arena import LSVMArena
from agt_server.agents.test_agents.lsvm.min_bidder.my_agent import MinBidAgent
from agt_server.agents.test_agents.lsvm.jump_bidder.jump_bidder import JumpBidder
from agt_server.agents.test_agents.lsvm.truthful_bidder.my_agent import TruthfulBidder
from path_utils import path_from_local_root
from uniform_policy import UniformPolicy

import time
import os
import random
import gzip
import json
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

NAME = "mulberry"
NUM_POSSIBLE_STATES = 216
NUM_POSSIBLE_ACTIONS = 8
INITIAL_STATE = 0
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.75
EXPLORATION_RATE = 0.2
TRAINING_MODE = True
SAVE_PATH_PREFIX = "q_table"
TIMESTAMP = time.strftime('%Y%m%d_%H%M%S')

class MyAgent(MyLSVMAgent):
    def __init__(self, name,
                 num_possible_states=NUM_POSSIBLE_STATES,
                 num_possible_actions=NUM_POSSIBLE_ACTIONS,
                 initial_state=INITIAL_STATE,
                 learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR,
                 exploration_rate=EXPLORATION_RATE,
                 training_mode=TRAINING_MODE, save_path_prefix=SAVE_PATH_PREFIX):
        self.num_possible_states = num_possible_states
        self.num_possible_actions = num_possible_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.training_mode = training_mode
        self.save_path_prefix = save_path_prefix
        self.s = initial_state
        self._reset_reward_tracker()
        super().__init__(name)

    def _reset_reward_tracker(self):
        # Allow tracker reset externally (ugly but it'll have to do)
        self.round_rewards = []
        self.auction_rewards = []
        self.auction_utility = []
        self.current_auction_reward = 0

    def _get_save_path(self):
        # Separate Q-tables for national and regional bidder
        return self.save_path_prefix + self.save_path_suffix + ".npy"
    
    def setup(self, restarts=20):
        # Set up separate Q-learning apparatus for national and regional bidders
        self.save_path_suffix = "_nat" if self.is_national_bidder() else "_reg"
        self.my_states = []
        self.training_policy = UniformPolicy(self.num_possible_actions)
        if self.save_path_prefix and os.path.isfile(self._get_save_path()):
            with open(self._get_save_path(), 'rb') as saved_q_table:
                self.q = np.load(saved_q_table)
                assert self.q.shape[0] == self.num_possible_states, "The Saved Q-Table has a different number of states than inputed, To train on the new states please delete the Saved Q-Table"
                assert self.q.shape[1] == self.num_possible_actions, "The number of possible actions in the saved file is different from the actual game, please delete and train again."
        else:
            # Initialize Q to random [-1, 1]
            self.q = np.array([[random.uniform(-1, 1)
                                for _ in range(self.num_possible_actions)]
                               for _ in range(self.num_possible_states)])
        # Begin with initial state and random action
        self.a = self.training_policy.get_move(self.s)
        self.s_prime = None

        # The grid is static so we calculate this once
        self.adj = self._get_adj() # Form {'A': ['G', 'B'], ...} (sanity-checked)

    # ---------------------------------------------------------
    # STRATEGIES
    # We have a grid of goods as follows:
    #     _   _   _   _   _   _ 
    #   | A | B | C | D | E | F |
    #     _   _   _   _   _   _ 
    #   | G | H | I | J | K | L |
    #     _   _   _   _   _   _ 
    #   | M | N | O | P | Q | R |
    #     _   _   _   _   _   _ 
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Conservative: Just MinBidder.
    # ---------------------------------------------------------
    def strategy_conservative(self):
        min_bids = self.get_min_bids()
        valuations = self.get_valuations() 
        bids = {} 
        for g in self.get_goods():
            if valuations[g] >= min_bids[g]:
                bids[g] = min_bids[g]
        return bids

    # ---------------------------------------------------------
    # Aggressive: Just JumpBidder.
    # ---------------------------------------------------------
    def strategy_aggressive(self, strength=0):
        min_bids = self.get_min_bids()
        valuations = self.get_valuations()
        bids = {}
        for g in valuations: 
            if valuations[g] > min_bids[g]:
                bids[g] = min_bids[g] + ((strength + 1) * 0.4 * (valuations[g] - min_bids[g]))
        return bids

    # ---------------------------------------------------------
    # Expansionist: We keep expanding the largest component in the
    # bundle by the cheapest tile until we reach a fixed threshold
    # size for the largest component.
    # ---------------------------------------------------------
    def _get_adj(self):
        # Adjacency list for goods
        goods = sorted(list(self.get_goods()))
        rows, cols = self.get_shape()
        adj = {g: [] for g in goods}
        for i, g in enumerate(goods):
            r, c = divmod(i, cols)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    adj[g].append(goods[nr * cols + nc])
        return adj

    def _get_components(self, current):
        # Components in current bundle ordered from largest to smallest
        seen = set()
        comps = []
        for g in current:
            if g in seen:
                continue
            seen.add(g)
            comp = set([g])
            q = deque([g])
            while q:
                cur = q.popleft()
                for nb in self.adj[cur]:
                    if nb in current and nb not in seen:
                        seen.add(nb)
                        comp.add(nb)
                        q.append(nb)
            comps.append(comp)
        comps.sort(key=lambda c: len(c), reverse=True)
        return comps

    def _get_frontier(self, comp, current):
        # Tiles adjacent to component but not in current bundle
        frontier = set()
        for g in comp:
            for nb in self.adj[g]:
                if nb not in current:
                    frontier.add(nb)
        return frontier

    def _grow_to_threshold(self, bids, required=5):
        min_bids = self.get_min_bids()
        
        # Get components in current bundle
        current = set(bids.keys())
        comps = self._get_components(current) # Form [{'M', 'G', 'N'}, {'F'}, ...] (sanity-checked)

        # Components already satisfy threshold
        if comps and len(comps[0]) >= required:
            return bids
        
        while True:
            comps = self._get_components(current)
            largest = comps[0] if comps else set()
            # Components now satisfy threshold
            if len(largest) >= required:
                return bids
            # Else expand largest component by cheapest tile
            frontier = self._get_frontier(largest, current) if largest else set()
            if frontier:
                cheapest = min(frontier, key=lambda g: min_bids[g])
                bids[cheapest] = min_bids[cheapest]
                current.add(cheapest)
            else:
                return bids

    def _grow_comps_national(self, bids):
        # "Extend bundle to ensure components of 9-13 tiles"
        return self._grow_to_threshold(bids, required=12)

    def _grow_comps_regional(self, bids):
        # "Extend bundle to ensure components of 3-7 tiles"
        return self._grow_to_threshold(bids, required=6)

    def strategy_expansionist(self):
        bids = self.strategy_conservative()
        if self.is_national_bidder():
            return self._grow_comps_national(bids)
        else:
            return self._grow_comps_regional(bids)

    # ---------------------------------------------------------
    # Connector: We identify goods that bridge two disconnected
    # components and bid on them even if they are a bit pricey.
    # ---------------------------------------------------------
    def _connect_comps(self, bids):
        min_bids = self.get_min_bids()
        
        # Get components in current bundle
        current = set(bids.keys())
        comps = self._get_components(current) # Form [{'M', 'G', 'N'}, {'F'}, ...] (sanity-checked)

        # No components to connect
        if len(comps) < 2:
            return bids

        # Label goods in bundle
        good_to_comp = {}
        for i, comp in enumerate(comps):
            for g in comp:
                good_to_comp[g] = i

        for g in self.get_goods():
            if g in current: continue
            nb_comp_is = set()
            for nb in self.adj[g]:
                if nb in good_to_comp:
                    nb_comp_is.add(good_to_comp[nb])
            # Add any bridging goods to bundle
            if len(nb_comp_is) >= 2:
                bids[g] = min_bids[g]
        return bids

    def strategy_connector(self):
        bids = self.strategy_conservative()
        return self._connect_comps(bids)

    # ---------------------------------------------------------
    # Focused: We prioritize maintaining components in specific
    # zones of the grid.
    # ---------------------------------------------------------
    def _get_indices(self):
        # Specific to our particular grid shape
        rows, cols = self.get_shape()
        indices = [
            [r*cols + c for r in range(rows) for c in range(0, 2)], # Left
            [r*cols + c for r in range(rows) for c in range(2, 4)], # Center
            [r*cols + c for r in range(rows) for c in range(4, 6)]  # Right
        ]
        return indices

    def strategy_focused(self, region):
        bids = self.strategy_conservative()

        # Which goods are we targetting?
        goods = sorted(list(self.get_goods())) # This allows us to do modular logic
        region_indices = self._get_indices()[region]
        target_goods = {goods[idx] for idx in region_indices}

        # Only bid if good is not too expensive
        min_bids = self.get_min_bids()
        valuations = self.get_valuations()
        for g in target_goods:
            if g not in bids and min_bids[g] / (valuations[g] + 0.01) < 2.5:
                bids[g] = min_bids[g]
        return bids

    # ---------------------------------------------------------
    # STATES
    # ---------------------------------------------------------
    def _f_phase(self):
        # "Termination point guaranteed to be at least 1000"
        phase_bins = [10, 100]
        return sum(self.get_current_round() > b for b in phase_bins)

    def _f_market_activity(self):
        # How much have the prices moved since the previous round?
        price_history = self.get_price_history_map()

        # No activity yet
        if len(price_history) < 2:
            return 0

        # Bin activity to prevent combinatorial explosion
        goods = self.get_goods()
        curr, prev = price_history[-1], price_history[-2]
        contest_bins = [2]
        contested = sum(1 for g in goods if curr[g] - prev[g] >= 0.1)
        return sum(contested > b for b in contest_bins)

    def _f_hottest_region(self):
        # Which region has the most activity?
        price_history = self.get_price_history_map()

        # No activity yet
        if len(price_history) < 2:
            return 1 # Default to centre

        # Changes are non-negative
        goods = sorted(list(self.get_goods())) # This allows us to do modular logic
        curr, prev = price_history[-1], price_history[-2]
        deltas = []
        for region_indices in self._get_indices():
            delta = sum(curr[goods[idx]] - prev[goods[idx]] for idx in region_indices)
            deltas.append(delta)

        # Break ties randomly
        max_val = np.max(deltas)
        candidates = np.flatnonzero(np.array(deltas) == max_val)
        return np.random.choice(candidates)

    def _f_region(self):
        # Which region is mine?
        regional_good = self.get_regional_good()
        if regional_good is not None:
            goods = sorted(list(self.get_goods())) # This allows us to do modular logic
            for region, region_indices in enumerate(self._get_indices()):
                for idx in region_indices:
                    if goods[idx] == regional_good:
                        return region
        return 1 # Default to centre

    def _f_allocation_size(self):
        # How much are we tentatively holding onto?
        tentative_alloc = self.get_tentative_allocation()
        alloc_bins = [4]
        return sum(len(tentative_alloc) > b for b in alloc_bins)

    def _f_price_ratio(self):
        # Under how much price pressure are we?
        total_prices = self.calc_total_prices()
        total_valuations = self.calc_total_valuation()
        price_ratio = total_prices / total_valuations if total_valuations > 0 else 0
        price_ratio_bins = [0.5]
        return sum(price_ratio > b for b in price_ratio_bins)

    def _encode_fs(self, fs):
        # Represent all features in a single integer
        bases = [3, 2, 3, 3, 2, 2] # 216 states
        idx = 0
        mult = 1
        for f, base in zip(fs, bases):
            idx += f * mult
            mult *= base
        return idx

    def determine_state(self):
        # Encode all features into a single integer
        fs = (
            self._f_phase(),
            self._f_market_activity(),
            self._f_hottest_region(),
            self._f_region(),
            self._f_allocation_size(),
            self._f_price_ratio()
        )
        return self._encode_fs(fs)

    # ---------------------------------------------------------
    # LEARNING
    # ---------------------------------------------------------
    def update_rule(self, reward):
        self.q[self.s, self.a] += self.learning_rate * \
            (reward + self.discount_factor *
             np.max(self.q[self.s_prime]) - self.q[self.s, self.a])

    def choose_next_move(self, s_prime):
        # In the next round, your agent will be in state [s_prime]. What move will it play?
        if (self.training_mode and random.random() < self.exploration_rate):
            return self.training_policy.get_move(self.s)
        else:
            return np.argmax(self.q[s_prime])
    
    def calculate_reward(self):
        # Change in utility per round (when update is called, previous_util has been updated)
        util_history = self.get_previous_util()
        if len(util_history) >= 2:
            return util_history[-1] - util_history[-2]
        return util_history[-1]

    # ---------------------------------------------------------
    # BIDDING
    # ---------------------------------------------------------
    def get_bids(self):
        # Map actions to strategies
        match self.a:
            case 0: bids = self.strategy_conservative()
            case 1: bids = self.strategy_aggressive(0)
            case 2: bids = self.strategy_aggressive(1)
            case 3: bids = self.strategy_expansionist()
            case 4: bids = self.strategy_connector()
            case 5: bids = self.strategy_focused(0)
            case 6: bids = self.strategy_focused(1)
            case 7: bids = self.strategy_focused(2)
        assert self.is_valid_bid_bundle(bids), "The proposed bid bundle is invalid. Change your strategies."
        return bids
    
    # ---------------------------------------------------------
    # WRAPPING UP
    # ---------------------------------------------------------
    def update(self):
        # Calculate and track reward
        reward = self.calculate_reward()
        self.round_rewards.append(reward)
        self.current_auction_reward += reward

        # Q-learning update and state transition
        self.s_prime = self.determine_state()
        self.update_rule(reward)
        self.a = self.choose_next_move(self.s_prime)
        self.s = self.s_prime
        self.s_prime = None 

    def teardown(self):
        # Track auction reward
        self.auction_utility.append(self.calc_total_utility())
        self.auction_rewards.append(self.current_auction_reward)
        self.current_auction_reward = 0

        # Only save here to boost performance
        if self.save_path_prefix:
            with open(self._get_save_path(), 'wb') as saved_q_table:
                np.save(saved_q_table, self.q)

################### SUBMISSION #####################
my_agent_submission = MyAgent(NAME)
####################################################

# ---------------------------------------------------------
# PROCESSING
# ---------------------------------------------------------
class Processor:
    def __init__(self):
        pass

    @staticmethod
    def process_saved_game(filepath): 
        # Here is some example code to load in a saved game in the format of a json.gz and to work with it
        print(f"Processing: {filepath}")
        
        # NOTE: Data is a dictionary mapping 
        with gzip.open(filepath, 'rt', encoding='UTF-8') as f:
            game_data = json.load(f)
            for agent, agent_data in game_data.items(): 
                if agent_data['valuations'] is not None: 
                    # agent is the name of the agent whose data is being processed 
                    agent = agent 
                    
                    # bid_history is the bidding history of the agent as a list of maps from good to bid
                    bid_history = agent_data['bid_history']
                    
                    # price_history is the price history of the agent as a list of maps from good to price
                    price_history = agent_data['price_history']
                    
                    # util_history is the history of the agent's previous utilities 
                    util_history = agent_data['util_history']
                    
                    # util_history is the history of the previous tentative winners of all goods as a list of maps from good to winner
                    winner_history = agent_data['winner_history']
                    
                    # elo is the agent's elo as a string
                    elo = agent_data['elo']
                    
                    # is_national_bidder is a boolean indicating whether or not the agent is a national bidder in this game 
                    is_national_bidder = agent_data['is_national_bidder']
                    
                    # valuations is the valuations the agent recieved for each good as a map from good to valuation
                    valuations = agent_data['valuations']
                    
                    # regional_good is the regional good assigned to the agent 
                    # This is None in the case that the bidder is a national bidder 
                    regional_good = agent_data['regional_good']
                
                # TODO: If you are planning on learning from previously saved games enter your code below. 

    @staticmethod
    def process_saved_dir(dirpath): 
        # Here is some example code to load in all saved game in the format of a json.gz in a directory and to work with it
        for filename in os.listdir(dirpath):
            if filename.endswith('.json.gz'):
                filepath = os.path.join(dirpath, filename)
                Processor.process_saved_game(filepath)

# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------
class Trainer:
    def __init__(self):
        pass

    @staticmethod
    def _plot_learning_curve(ys, xlabel, ylabel, title, fname):
        # Visualise non-smoothed
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ys, alpha=0.6, linewidth=1, label="raw")

        # Use smoothing of 50 if enough data
        window = 100
        if len(ys) >= window:
            kernel = np.ones(window) / window
            ys_smooth = np.convolve(ys, kernel, mode="same")
            ax.plot(ys_smooth, alpha=0.8, linewidth=1, label="smoothed")

        # Annotate and save
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"fig/{TIMESTAMP}_{fname}.png", dpi=300, bbox_inches="tight")
        print(f"Saved fig/{TIMESTAMP}_{fname}.png")

    @staticmethod
    def _plot_learning_curves(agent, num_cycles_per_player, phase=0):
        # Per-auction utility
        auction_utility = np.array(agent.auction_utility)
        Trainer._plot_learning_curve(auction_utility, "Auction", "Reward",
                                     f"Utility per auction across {num_cycles_per_player} cycles", f"{phase}_auction_utility")

        # Per-auction cumulative reward
        auction_rewards = np.array(agent.auction_rewards)
        Trainer._plot_learning_curve(auction_rewards, "Auction", "Reward",
                                     f"Reward per auction across {num_cycles_per_player} cycles", f"{phase}_auction_rewards")

        # Per-round reward
        round_rewards = np.array(agent.round_rewards)
        Trainer._plot_learning_curve(round_rewards, "Round", "Reward",
                                     f"Reward per round across {num_cycles_per_player} cycles", f"{phase}_round_rewards")

    @staticmethod
    def _print_policy_change(initial_policy, final_policy, phase=0):
        # Calculate and log policy changes
        policy_change = np.sum(initial_policy != final_policy)
        change_percentage = (policy_change / NUM_POSSIBLE_STATES) * 100
        
        # Which strategies changed most?
        changed_from = initial_policy[initial_policy != final_policy]
        changed_to = final_policy[initial_policy != final_policy]
        
        # Save to plaintext file
        with open(f"log/{TIMESTAMP}_policy_changes.txt", 'a') as f:
            f.write(f"=== Phase {phase} ===\n")
            f.write(f"Policy changes: {policy_change}/{NUM_POSSIBLE_STATES} ({change_percentage:.1f}%)\n")
            f.write(f"Change distribution:\n")
            
            # Track transitions between strategies
            if len(changed_from) > 0:
                transition_counts = {}
                for i, j in zip(changed_from, changed_to):
                    transition_counts[(i, j)] = transition_counts.get((i, j), 0) + 1
                
                # Write top 5 transitions
                for (from_s, to_s), count in sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    strategies = ["Conservative", "Aggressive0", "Aggressive1", "Expansionist", 
                                  "Connector", "FocusedL", "FocusedC", "FocusedR"]
                    f.write(f"  {strategies[from_s]} > {strategies[to_s]}: {count} states\n")
            f.write("\n")

    @staticmethod
    def _print_q_change(initial_q, final_q, phase=0):
        # Calculate and log Q-table changes
        q_change = np.mean(np.abs(final_q - initial_q))
        q_improvement = np.mean(final_q - initial_q)  # Positive if improving
        
        # Calculate state-level statistics
        state_improvements = np.mean(final_q - initial_q, axis=1)
        improving_states = np.sum(state_improvements > 0)

        # Save to plaintext file
        with open(f"log/{TIMESTAMP}_q_changes.txt", 'a') as f:
            f.write(f"=== Phase {phase} ===\n")
            f.write(f"Average absolute Q-change: {q_change:.4f}\n")
            f.write(f"Average Q-improvement: {q_improvement:.4f}\n")
            f.write(f"States with Q-improvement: {improving_states}/{initial_q.shape[0]}\n")
            f.write(f"Q-value spread (final_q): {np.max(final_q) - np.min(final_q):.3f}\n")
            
            # Top five states with biggest improvement
            if improving_states > 0:
                top_improvements = np.argsort(state_improvements)[-5:][::-1]
                f.write("Top improving states:\n")
                for state in top_improvements:
                    f.write(f"  State {state}: +{state_improvements[state]:.3f}\n")
            f.write("\n")

    @staticmethod
    def run():
        # Run training and generate learning curves
        # 0. Create arena with one learner and everybody else fixed
        agent = MyAgent("MyAgent", training_mode=True)
        # Instead of mixed opponents, create a curriculum
        opponents_list = [
            # Phase 1. MinBidder
            (15, [MinBidAgent(f"MinBidder{i}") for i in range(5)]),

            # Phase 2. MinBidder + JumpBidder
            (15, [MinBidAgent(f"MinBidder{i}") for i in range(2)]
            + [JumpBidder(f"JumpBidder{i}") for i in range(3)]),
            
            # Phase 3. MinBidder + JumpBidder + TruthfulBidder
            (25, [MinBidAgent(f"MinBidder{i}") for i in range(1)]
            + [JumpBidder(f"JumpBidder{i}") for i in range(2)]
            + [TruthfulBidder(f"TruthfulBidder{i}") for i in range(2)]),
            
            # Phase 4. Self-play
            (100, [MyAgent(f"MyAgent{i}", training_mode=False) for i in range(5)]),

            # Phase 5: All hiccups break loose
            (100, [MyAgent(f"MyAgent{i}", training_mode=False) for i in range(2)]
            + [MinBidAgent("MinBidder1"), JumpBidder("JumpBidder1"), TruthfulBidder("TruthfulBidder1")])
        ]

        # 1. Initialize Q-table
        arena = LSVMArena(
            num_cycles_per_player=1,
            players=[agent] + [MinBidAgent(f"MinBidder{i}") for i in range(6)]
        )
        arena.run()
        q_0 = agent.q.copy()
        policy_0 = np.argmax(q_0, axis=1)

        # 2. Train in phases
        for phase, (num_cycles, opponents) in enumerate(opponents_list):
            print(f"\n=== Training Phase {phase+1} ===")
            # Train against current opponents
            arena = LSVMArena(
                num_cycles_per_player=num_cycles,
                players=[agent] + opponents
            )
            arena.run()

            # Compare Q-tables and policies
            q_1 = agent.q.copy()
            policy_1 = np.argmax(q_1, axis=1)
            Trainer._print_q_change(q_0, q_1, phase)
            Trainer._print_policy_change(policy_0, policy_1, phase)

            # Plot learning curves
            Trainer._plot_learning_curves(agent, num_cycles, phase)
            agent._reset_reward_tracker() # So that the curves are fresh next phase

        # This is our final agent
        return agent

if __name__ == "__main__":
    # Train agent against fixed opponents
    trained_agent = Trainer.run()

    # Test agent against fixed opponents
    arena = LSVMArena(
        num_cycles_per_player=3,
        timeout=1,
        players=[MyAgent(f"MyAgent{i}", training_mode=False) for i in range(4)]
                + [MinBidAgent("MinBidder1"), JumpBidder("JumpBidder1"), TruthfulBidder("TruthfulBidder1")]
    )
    arena.run()
    exit()
    # Heres an example of how to process a singular file 
    # process_saved_game(path_from_local_root("saved_games/2024-04-08_17-36-34.json.gz"))
    # or every file in a directory 
    # process_saved_dir(path_from_local_root("saved_games"))
    
    ### DO NOT TOUCH THIS #####
    agent = MyAgent(NAME)
    arena = LSVMArena(
        num_cycles_per_player = 3,
        timeout=1,
        local_save_path="saved_games",
        players=[
            agent,
            MyAgent("CP - MyAgent"),
            MyAgent("CP2 - MyAgent"),
            MyAgent("CP3 - MyAgent"),
            MinBidAgent("Min Bidder"), 
            JumpBidder("Jump Bidder"), 
            TruthfulBidder("Truthful Bidder"), 
        ]
    )
    
    start = time.time()
    arena.run()
    end = time.time()
    print(f"{end - start} Seconds Elapsed")
