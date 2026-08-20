import unittest
import numpy as np
import gymnasium as gym

from engine.environments.ulti import UltiEnv

class TestUltiEnv(unittest.TestCase):
    def setUp(self):
        self.env = UltiEnv()

    def test_spaces(self):
        self.assertIsInstance(self.env.action_space, gym.spaces.Discrete)
        self.assertEqual(self.env.action_space.n, 54)
        
        self.assertIsInstance(self.env.observation_space, gym.spaces.Dict)
        self.assertIn("hand", self.env.observation_space.spaces)
        self.assertIn("trick_history", self.env.observation_space.spaces)
        self.assertIn("deduction_flags", self.env.observation_space.spaces)

    def test_reset(self):
        obs, info = self.env.reset()
        
        self.assertIn("hand", obs)
        self.assertIn("trick_history", obs)
        self.assertIn("deduction_flags", obs)
        
        self.assertEqual(obs["hand"].shape, (32,))
        self.assertEqual(obs["trick_history"].shape, (30,))
        self.assertEqual(obs["deduction_flags"].shape, (12,))
        
        self.assertIn("action_mask", info)
        self.assertEqual(info["action_mask"].shape, (54,))
        self.assertEqual(self.env.phase, "drop_talon")

    def test_step_bidding(self):
        obs, info = self.env.reset()
        
        # Test a valid bid action
        # 0 is always pass, which is a valid bid
        valid_action = 0
        self.assertTrue(info["action_mask"][valid_action])
        
        next_obs, reward, terminated, truncated, next_info = self.env.step(valid_action)
        
        self.assertIsInstance(next_obs, dict)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("action_mask", next_info)
        
    def test_full_game_random_rollout(self):
        obs, info = self.env.reset()
        
        terminated = False
        truncated = False
        
        # We cap at a large number to avoid infinite loops in case of bugs
        steps = 0
        while not (terminated or truncated) and steps < 200:
            mask = info["action_mask"]
            valid_actions = np.where(mask)[0]
            self.assertTrue(len(valid_actions) > 0, "No valid actions available")
            
            # Choose a random valid action
            action = np.random.choice(valid_actions)
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            steps += 1
            
        self.assertTrue(terminated)
        self.assertLess(steps, 200, "Game did not terminate in a reasonable number of steps")

if __name__ == '__main__':
    unittest.main()
