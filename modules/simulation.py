"""
Simulation Module
Agent-based simulation for business location analysis
"""

import random
import numpy as np


class Agent:
    """
    Represents a potential customer/visitor agent with income and preferences.
    """
    
    def __init__(self, income: float, preference: dict = None):
        """
        Initialize an Agent.
        
        Args:
            income: Agent's income level (0-100)
            preference: Dictionary of business type preferences
        """
        self.income = income
        self.preference = preference or {}
    
    def get_preference(self, business_type: str) -> float:
        """
        Get preference score for a business type.
        
        Args:
            business_type: Type of business
            
        Returns:
            Preference score (0-1)
        """
        return self.preference.get(business_type, 0.5)


def run_simulation(location: tuple, agents: list, businesses: list, 
                  max_steps: int = 100) -> dict:
    """
    Run agent-based simulation to predict business viability at a location.
    
    Args:
        location: (lat, lon) coordinate
        agents: List of Agent objects
        businesses: List of business types to simulate
        max_steps: Number of simulation steps
        
    Returns:
        Dictionary with simulation results
    """
    if not agents or not businesses:
        return {
            "location": location,
            "total_attraction": 0,
            "business_scores": {},
            "average_satisfaction": 0,
            "predicted_revenue": 0
        }
    
    results = {
        "location": location,
        "business_scores": {},
        "agent_attractions": [],
        "total_attraction": 0,
        "average_satisfaction": 0
    }
    
    # Simulate for each business type
    for business in businesses:
        attraction_sum = 0
        satisfaction_sum = 0
        
        for step in range(max_steps):
            step_attraction = 0
            
            for agent in agents:
                # Calculate attraction score
                preference_score = agent.get_preference(business)
                income_factor = agent.income * 0.01  # Normalize income
                proximity_factor = random.uniform(0.8, 1.0)  # Random proximity factor
                
                # Combine factors
                attraction = (
                    preference_score * 0.4 +
                    income_factor * 0.3 +
                    proximity_factor * 0.3
                )
                
                step_attraction += attraction
                satisfaction_sum += attraction
            
            attraction_sum += step_attraction
        
        avg_attraction = attraction_sum / max_steps if max_steps > 0 else 0
        avg_satisfaction = satisfaction_sum / (max_steps * len(agents)) if (max_steps * len(agents)) > 0 else 0
        
        results["business_scores"][business] = {
            "total_attraction": avg_attraction,
            "satisfaction_score": avg_satisfaction,
            "viability": "high" if avg_attraction > 50 else "medium" if avg_attraction > 25 else "low"
        }
        
        results["total_attraction"] += avg_attraction
    
    # Calculate averages
    num_businesses = len(businesses)
    if num_businesses > 0:
        results["average_satisfaction"] = results["total_attraction"] / num_businesses
        results["predicted_revenue"] = results["total_attraction"] * 10  # Simple revenue estimate
    
    return results


def generate_agents(population_size: int = 100, income_distribution: str = 'normal') -> list:
    """
    Generate a population of agents for simulation.
    
    Args:
        population_size: Number of agents to generate
        income_distribution: Distribution type ('normal', 'uniform', 'skewed')
        
    Returns:
        List of Agent objects
    """
    agents = []
    
    for i in range(population_size):
        # Generate income based on distribution
        if income_distribution == 'normal':
            income = np.random.normal(50, 15)
        elif income_distribution == 'uniform':
            income = np.random.uniform(0, 100)
        else:  # skewed
            income = np.random.gamma(shape=2, scale=20)
        
        income = max(0, min(100, income))  # Clamp to 0-100
        
        # Generate random preferences
        business_types = ['cafe', 'restaurant', 'shop', 'supermarket', 'bank', 'gym']
        preference = {biz: random.uniform(0, 1) for biz in business_types}
        
        agent = Agent(income, preference)
        agents.append(agent)
    
    return agents


def analyze_simulation_results(results: dict) -> str:
    """
    Generate a human-readable analysis of simulation results.
    
    Args:
        results: Dictionary from run_simulation()
        
    Returns:
        Formatted analysis string
    """
    analysis = []
    analysis.append("=" * 60)
    analysis.append("AGENT-BASED SIMULATION RESULTS")
    analysis.append("=" * 60)
    
    analysis.append(f"Location: {results.get('location', 'Unknown')}")
    analysis.append(f"Average Satisfaction: {results.get('average_satisfaction', 0):.2f}")
    analysis.append(f"Predicted Revenue (relative): ${results.get('predicted_revenue', 0):.2f}")
    analysis.append("")
    
    analysis.append("Business Type Analysis:")
    for business, scores in results.get('business_scores', {}).items():
        analysis.append(f"  • {business.upper()}")
        analysis.append(f"    - Attraction Score: {scores.get('total_attraction', 0):.2f}")
        analysis.append(f"    - Satisfaction: {scores.get('satisfaction_score', 0):.2f}")
        analysis.append(f"    - Viability: {scores.get('viability', 'unknown')}")
    
    analysis.append("=" * 60)
    
    return "\n".join(analysis)
