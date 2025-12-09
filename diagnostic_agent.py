from agt_server.local_games.lsvm_arena import LSVMArena
from agt_server.agents.test_agents.lsvm.min_bidder.my_agent import MinBidAgent
from agt_server.agents.test_agents.lsvm.jump_bidder.jump_bidder import JumpBidder
from agt_server.agents.test_agents.lsvm.truthful_bidder.my_agent import TruthfulBidder
from my_agent import MyAgent

import numpy as np
import time

TIMESTAMP = time.strftime('%Y%m%d_%H%M%S')

class DiagnosticAgent(MyAgent):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.feature_records = []
        self.state_records = []
        
    # ---------------------------------------------------------
    # STATES
    # ---------------------------------------------------------
    def determine_state(self):
        # Compute individual features
        phase = self._f_phase()
        market_activity = self._f_market_activity()
        hottest_region = self._f_hottest_region()
        my_region = self._f_region()
        allocation_size = self._f_allocation_size()
        price_ratio = self._f_price_ratio()
        
        # Record features for analysis
        feature_tuple = (
            phase,
            market_activity,
            hottest_region,
            my_region,
            allocation_size,
            price_ratio
        )
        self.feature_records.append(feature_tuple)
        
        # Also record the encoded state
        encoded_state = self._encode_fs(feature_tuple)
        self.state_records.append(encoded_state)
        
        return encoded_state

    # ---------------------------------------------------------
    # DIAGNOSTICS
    # ---------------------------------------------------------
    def _get_base(self, feature_idx):
        # Helper to get base for each feature
        bases = [3, 2, 3, 3, 2, 2]
        return bases[feature_idx]
    
    def print_feature_stats(self, filepath=None):
        # Print feature distribution statistics
        features = list(zip(*self.feature_records))
        feature_names = [
            "Phase",
            "Market activity",
            "Hottest region", 
            "My region",
            "Allocation size",
            "Price ratio"
        ]

        with open(filepath, 'a') as f:
            f.write(f"Total observations: {len(self.feature_records)}\n")
            
            # Feature analysis
            for i, (name, values) in enumerate(zip(feature_names, features)):
                unique, counts = np.unique(values, return_counts=True)
                percentages = (counts / len(values) * 100)
                
                f.write(f"\n{name} (Feature {i}):\n")
                f.write(f"  Possible values: {list(range(self._get_base(i)))}\n")
                f.write("  Distribution:\n")
                for val, count, pct in zip(unique, counts, percentages):
                    f.write(f"    {val}: {count} observations ({pct:.1f}%)\n")
            
            # State analysis
            unique_states = len(np.unique(self.state_records))
            state_counts = np.bincount(self.state_records, minlength=self.num_possible_states)
            f.write(f"\nUnique states observed: {unique_states}/{self.num_possible_states}\n")
            f.write(f"\nState occupancy (top 10):\n")
            for state in np.argsort(state_counts)[-10:][::-1]:
                count = state_counts[state]
                if count > 0:
                    pct = (count / len(self.state_records) * 100)
                    f.write(f"  State {state}: {count} ({pct:.1f}%)\n")

# ---------------------------------------------------------
# DIAGNOSTICS
# ---------------------------------------------------------
class Diagnostics:
    def __init__():
        pass

    @staticmethod
    def run(num_cycles=3, filepath=f"log/{TIMESTAMP}_diagnostics.txt"):
        # Run diagnostic test and analyse feature distributions
        diagnostic_agent = DiagnosticAgent("DiagnosticAgent", training_mode=False)
        
        # Create and run arena with fixed opponents
        arena = LSVMArena(
            num_cycles_per_player=num_cycles,
            timeout=1,
            players=[diagnostic_agent]
                    + [MyAgent(f"MyAgent{i}", training_mode=False) for i in range(3)]
                    + [MinBidAgent("MinBidder1"), JumpBidder("JumpBidder1"), TruthfulBidder("TruthfulBidder1")]
        )
        arena.run()

        # Save feature analysis
        diagnostic_agent.print_feature_stats(filepath=filepath)

if __name__ == "__main__":
    # Analyse feature distributions
    Diagnostics.run(num_cycles=3)
