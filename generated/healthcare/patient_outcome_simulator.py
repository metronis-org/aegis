"""
Auto-generated Simulator: patient_outcome_simulator
Domain: healthcare
Type: rl_environment
Dynamics: stochastic
"""

from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np
from gym import spaces


class PatientOutcomeSimulator(gym.Env):
    """
    rl_environment for healthcare domain.

    State Space: {'type': 'dict', 'fields': ['patient_vitals', 'current_medications', 'diagnoses']}
    Action Space: {'type': 'discrete', 'actions': ['prescribe_medication', 'order_test', 'refer_specialist', 'discharge']}
    Reward Function: patient_outcome_90_day
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the simulator."""
        super().__init__()
        self.config = config or {}

        # Define state space
        self.state_space_def = {
            "type": "dict",
            "fields": ["patient_vitals", "current_medications", "diagnoses"],
        }
        self.observation_space = self._create_observation_space()

        # Define action space
        self.action_space_def = {
            "type": "discrete",
            "actions": [
                "prescribe_medication",
                "order_test",
                "refer_specialist",
                "discharge",
            ],
        }
        self.action_space = self._create_action_space()

        # Initialize state
        self.state = None
        self.steps = 0
        self.max_steps = self.config.get("max_steps", 1000)

        # Dynamics model
        self.dynamics = "stochastic"
        self.transition_model = None  # Will be learned from data

    def _create_observation_space(self) -> spaces.Space:
        """Create Gym observation space from state_space_def."""
        # This is a simplified version - should be customized per domain

        space_dict = {}

        # type: dict

        space_dict["type"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        # fields: ['patient_vitals', 'current_medications', 'diagnoses']

        space_dict["fields"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

        return spaces.Dict(space_dict)

    def _create_action_space(self) -> spaces.Space:
        """Create Gym action space from action_space_def."""
        # Simplified - should be customized

        space_dict = {}

        # type: discrete
        space_dict["type"] = spaces.Discrete(2)  # Placeholder

        # actions: ['prescribe_medication', 'order_test', 'refer_specialist', 'discharge']
        space_dict["actions"] = spaces.Discrete(2)  # Placeholder

        return spaces.Dict(space_dict)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Any, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Initialize state
        self.state = self._get_initial_state()
        self.steps = 0

        info = {"episode": 0}
        return self.state, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.steps += 1

        # Transition to next state
        next_state = self._transition(self.state, action)

        # Compute reward
        reward = self._compute_reward(self.state, action, next_state)

        # Check if episode is done
        terminated = self._is_terminal(next_state)
        truncated = self.steps >= self.max_steps

        # Update state
        self.state = next_state

        # Additional info
        info = {
            "steps": self.steps,
            "reward": reward,
        }

        return self.state, reward, terminated, truncated, info

    def _get_initial_state(self) -> Dict[str, Any]:
        """Generate initial state."""

        # Initialize patient state
        return {
            "vitals": self._sample_vitals(),
            "labs": self._sample_labs(),
            "medications": [],
            "diagnoses": [],
            "demographics": self._sample_demographics(),
        }

    def _transition(self, state: Dict[str, Any], action: Any) -> Dict[str, Any]:
        """
        Transition function: s' = T(s, a)

        For stochastic dynamics, this samples from P(s'|s,a)
        """
        next_state = state.copy()

        # Add stochasticity to transitions
        noise = np.random.normal(0, 0.1, size=len(state))

        # Patient response to treatment
        if "prescribe_medication" in action:
            med = action["prescribe_medication"]
            next_state["medications"].append(med)

            # Simulate patient response (simplified)
            response_rate = self._get_treatment_response_rate(med, state)
            if np.random.random() < response_rate:
                # Improve vitals
                next_state["vitals"] = self._improve_vitals(state["vitals"])
            else:
                # May worsen or stay same
                next_state["vitals"] = self._worsen_vitals(
                    state["vitals"], probability=0.1
                )

        return next_state

    def _compute_reward(
        self, state: Dict[str, Any], action: Any, next_state: Dict[str, Any]
    ) -> float:
        """
        Reward function: r = R(s, a, s')

        Domain-specific reward shaping
        """
        reward = 0.0

        # Reward: patient_outcome_90_day
        # Patient outcome improvement
        vitals_improved = self._check_vitals_improvement(
            state.get("vitals", {}), next_state.get("vitals", {})
        )
        reward += 10.0 if vitals_improved else -5.0

        # Penalize adverse events
        if self._has_adverse_event(next_state):
            reward -= 50.0

        # Penalize unnecessary treatments
        if len(next_state["medications"]) > len(state["medications"]) + 1:
            reward -= 5.0  # Polypharmacy penalty

        return reward

    def _is_terminal(self, state: Dict[str, Any]) -> bool:
        """Check if episode should terminate."""

        # Episode ends if patient discharged or deceased
        if state.get("discharged", False):
            return True
        if self._patient_deceased(state):
            return True

        return False

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == "human":
            print(f"Step: {self.steps}")
            print(f"State: {self.state}")
        elif mode == "rgb_array":
            # Return RGB array for video recording
            return np.zeros((400, 600, 3), dtype=np.uint8)
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass

    # Helper methods (domain-specific)

    def _sample_vitals(self) -> Dict[str, float]:
        """Sample realistic vital signs."""
        return {
            "blood_pressure_systolic": np.random.normal(120, 15),
            "blood_pressure_diastolic": np.random.normal(80, 10),
            "heart_rate": np.random.normal(75, 10),
            "temperature": np.random.normal(98.6, 0.5),
            "respiratory_rate": np.random.normal(16, 2),
            "oxygen_saturation": np.random.normal(98, 1.5),
        }

    def _sample_labs(self) -> Dict[str, float]:
        """Sample lab values."""
        return {
            "glucose": np.random.normal(100, 20),
            "hemoglobin": np.random.normal(14, 2),
            "creatinine": np.random.normal(1.0, 0.3),
        }

    def _sample_demographics(self) -> Dict[str, Any]:
        """Sample patient demographics."""
        return {
            "age": np.random.randint(18, 90),
            "sex": np.random.choice(["M", "F"]),
            "weight": np.random.normal(70, 15),
            "height": np.random.normal(170, 10),
        }


# Register environment
gym.register(
    id="patient_outcome_simulator-v0",
    entry_point="PatientOutcomeSimulator",
    max_episode_steps=1000,
)
