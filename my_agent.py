from agt_server.agents.base_agents.lsvm_agent import MyLSVMAgent
from agt_server.local_games.lsvm_arena import LSVMArena
from agt_server.agents.test_agents.lsvm.min_bidder.my_agent import MinBidAgent
from agt_server.agents.test_agents.lsvm.jump_bidder.jump_bidder import JumpBidder
from agt_server.agents.test_agents.lsvm.truthful_bidder.my_agent import TruthfulBidder
import time
import os
import random
import gzip
import json
from path_utils import path_from_local_root

import numpy as np
from collections import deque
from uniform_policy import UniformPolicy


NAME = "mulberry"
NUM_POSSIBLE_STATES = 144
NUM_POSSIBLE_ACTIONS = 7
INITIAL_STATE = 0
LEARNING_RATE = 0.05
DISCOUNT_FACTOR = 0.90
EXPLORATION_RATE = 0.1
TRAINING_MODE = False
SAVE_PATH_PREFIX = "qtab3"

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
        super().__init__(name)

    def _get_save_path(self):
        # Separate Q-tables for national and regional bidder
        save_path_suffix = "_nat" if self.is_national_bidder() else "_reg"
        return self.save_path_prefix + save_path_suffix + ".npy"
    
    def setup(self, restarts=20):
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

        # The grid is static, so we calculate this once
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
    def strategy_aggressive(self):
        min_bids = self.get_min_bids()
        valuations = self.get_valuations()
        bids = {}
        for g in valuations: 
            if valuations[g] > min_bids[g]:
                bids[g] = random.uniform(min_bids[g], valuations[g])
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
        valuations = self.get_valuations() 
        
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
            if len(nb_comp_is) >= 2 and min_bids[g] / valuations[g] < 1.5: # TODO: Should this be more lenient?
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
        tentative_alloc = self.get_tentative_allocation()
        for g in target_goods:
            if g not in bids and min_bids[g] / valuations[g] < 1.5: # TODO: Should this be more lenient?
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
        contest_bins = [0]
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
        return np.argmax(deltas) # TODO: Tie-breaking

    def _f_competition(self):
        # How many unique winners were there?
        # TODO: Find out what in the arena code causes the below exception
        try: unique_winners = np.unique(self.get_previous_winners())
        except: unique_winners = []
        competition_bins = [2] # TODO: Should we change this to one?
        return sum(len(unique_winners) > b for b in competition_bins)

    def _f_allocation_size(self):
        # How much are we tentatively holding onto?
        tentative_alloc = self.get_tentative_allocation()
        alloc_bins = [5]
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
        bases = [3, 2, 3, 2, 2, 2] # 144 states
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
            self._f_competition(),
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
    
    # ---------------------------------------------------------
    # BIDDING
    # ---------------------------------------------------------
    def get_bids(self):
        # Map actions to strategies
        match self.a:
            case 0: bids = self.strategy_conservative()
            case 1: bids = self.strategy_aggressive()
            case 2: bids = self.strategy_expansionist()
            case 3: bids = self.strategy_connector()
            case 4: bids = self.strategy_focused(0)
            case 5: bids = self.strategy_focused(1)
            case 6: bids = self.strategy_focused(2)
        assert self.is_valid_bid_bundle(bids), "The proposed bid bundle is invalid. Change your strategies."
        return self.clip_bids(bids)
    
    # ---------------------------------------------------------
    # WRAPPING UP
    # ---------------------------------------------------------
    def update(self):
        self.s_prime = self.determine_state()
        previous_util = self.get_previous_util()[-1] # Given its tentative allocation in the previous round of the auction
        self.update_rule(previous_util)
        self.s = self.s_prime
        self.a = self.choose_next_move(self.s_prime)
        self.s_prime = None 

    def teardown(self):
        # Only save here to boost performance
        if self.save_path_prefix:
            with open(self._get_save_path(), 'wb') as saved_q_table:
                np.save(saved_q_table, self.q)

################### SUBMISSION #####################
my_agent_submission = MyAgent(NAME)
####################################################


def process_saved_game(filepath): 
    """ 
    Here is some example code to load in a saved game in the format of a json.gz and to work with it
    """
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
            
            
        
def process_saved_dir(dirpath): 
    """ 
     Here is some example code to load in all saved game in the format of a json.gz in a directory and to work with it
    """
    for filename in os.listdir(dirpath):
        if filename.endswith('.json.gz'):
            filepath = os.path.join(dirpath, filename)
            process_saved_game(filepath)
            

if __name__ == "__main__":
    
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
