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
NUM_POSSIBLE_STATES = 1 # Currently a bandit
NUM_POSSIBLE_ACTIONS = 2
INITIAL_STATE = 0
LEARNING_RATE = 0.05
DISCOUNT_FACTOR = 0.90
EXPLORATION_RATE = 0.05
TRAINING_MODE = False
SAVE_PATH_PREFIX = "qtable"

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
        for good in self.get_goods():
            if valuations[good] >= min_bids[good]:
                bids[good] = valuations[good]
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

    def _get_components(self, current, adj):
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
                for nb in adj[cur]:
                    if nb in current and nb not in seen:
                        seen.add(nb)
                        comp.add(nb)
                        q.append(nb)
            comps.append(comp)
        comps.sort(key=lambda c: len(c), reverse=True)
        return comps

    def _get_frontier(self, comp, current, adj):
        # Tiles adjacent to component but not in current bundle
        frontier = set()
        for g in comp:
            for nb in adj[g]:
                if nb not in current:
                    frontier.add(nb)
        return frontier

    def _grow_to_threshold(self, bids, required=5):
        # TODO: Think about possible improvements
        min_bids = self.get_min_bids()
        adj = self._get_adj() # Form {'A': ['G', 'B'], ...} (sanity-checked)
        
        # Get components in current bundle
        current = set(bids.keys())
        comps = self._get_components(current, adj) # Form [{'M', 'G', 'N'}, {'F'}, ...] (sanity-checked)

        # Components already satisfy threshold
        if comps and len(comps[0]) >= required:
            return bids
        
        while True:
            comps = self._get_components(current, adj)
            largest = comps[0] if comps else set()
            # Components now satisfy threshold
            if len(largest) >= required:
                return bids
            # Else expand largest component by cheapest tile
            frontier = self._get_frontier(largest, current, adj) if largest else set()
            if frontier:
                cheapest = min(frontier, key=lambda g: min_bids[g])
                bids[cheapest] = min_bids[cheapest]
                current.add(cheapest)
            else:
                return bids

    def _connect_comps_national(self, bids):
        # Extend bundle to ensure components of 9-13 tiles
        return self._grow_to_threshold(bids, required=9)

    def _connect_comps_regional(self, bids):
        # Extend bundle to ensure components of 3-7 tiles
        return self._grow_to_threshold(bids, required=3)

    def strategy_expansionist(self):
        bids = self.strategy_conservative()
        if self.is_national_bidder():
            return self._connect_comps_national(bids)
        else:
            return self._connect_comps_regional(bids)

    # ---------------------------------------------------------
    # LEARNING
    # ---------------------------------------------------------
    def determine_state(self):
        curr_state = 0
        # TODO: Develop states (currently a bandit)
        return curr_state

    def update_rule(self, reward):
        self.q[self.s, self.a] += self.learning_rate * \
            (reward + self.discount_factor *
             np.max(self.q[self.s_prime]) - self.q[self.s, self.a])
        if self.save_path_prefix:
            with open(self._get_save_path(), 'wb') as saved_q_table:
                np.save(saved_q_table, self.q)

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
        # TODO: Develop more strategies
        match self.a:
            case 0:
                return self.strategy_conservative()
            case 1:
                return self.strategy_expansionist()
    
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
        pass

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
