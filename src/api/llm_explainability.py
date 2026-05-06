import json

class PricingExplainabilityAgent:
    def __init__(self):
        # In a production environment, this would initialize a connection to 
        # OpenAI, Anthropic, or SAP Generative AI Hub via API keys.
        self.system_prompt = (
            "You are an AI Pricing Analyst. Explain the dynamic price adjustment "
            "based on the provided market context in 2 short, professional sentences."
        )

    def generate_explanation(self, sku_id, old_price, new_price, demand_score, inventory_ratio, comp_delta):
        """
        Simulates an LLM call to generate a natural language explanation for the dynamic price change.
        """
        direction = "increased" if new_price > old_price else "decreased"
        pct_change = abs((new_price - old_price) / max(old_price, 1)) * 100
        
        # Rule-based simulation of what an LLM would output based on context
        if demand_score > 300 and inventory_ratio < 0.3:
            reasoning = (
                f"The price for {sku_id} was {direction} by {pct_change:.1f}% to balance exceptionally high demand "
                f"against critically low inventory levels ({(inventory_ratio*100):.0f}% remaining). "
                f"This proactive adjustment prevents imminent stockouts while optimizing revenue capture during the demand surge."
            )
        elif comp_delta < -0.05:
            reasoning = (
                f"We {direction} the price of {sku_id} by {pct_change:.1f}% to directly address aggressive competitor actions. "
                f"Key competitors recently dropped their prices by {abs(comp_delta*100):.1f}%, and this adjustment ensures we retain our market share while respecting margin floors."
            )
        elif inventory_ratio > 0.8 and demand_score < 100:
            reasoning = (
                f"The price was strategically {direction} by {pct_change:.1f}% to stimulate sales velocity for slow-moving stock. "
                f"With inventory sitting at {(inventory_ratio*100):.0f}%, this promotional pricing prevents capital lock-up and reduces holding costs."
            )
        else:
            reasoning = (
                f"The algorithm gently {direction} the price of {sku_id} by {pct_change:.1f}% based on routine predictive modeling. "
                f"Current inventory and demand signals are highly stable, allowing for slight margin optimization without risking conversion drop-offs."
            )
            
        return {
            "sku": sku_id,
            "old_price": round(old_price, 2),
            "new_price": round(new_price, 2),
            "explanation": reasoning,
            "llm_confidence": 0.94
        }

if __name__ == "__main__":
    # Test the LLM Explainability Agent
    agent = PricingExplainabilityAgent()
    
    # Simulate a scenario: High demand, very low inventory
    result = agent.generate_explanation("olist_prod_123", 100.0, 125.0, 450, 0.15, 0.0)
    print("--- Scenario 1: Surge Pricing ---")
    print(json.dumps(result, indent=2))
    
    # Simulate a scenario: Competitor undercutting
    result2 = agent.generate_explanation("olist_prod_999", 50.0, 45.0, 150, 0.60, -0.10)
    print("\n--- Scenario 2: Competitor Match ---")
    print(json.dumps(result2, indent=2))
