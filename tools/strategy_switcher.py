# tools/strategy_switcher.py

def select_strategy(context: dict) -> dict:
    """
    Takes in market context and returns a suitable strategy.
    
    Args:
        context (dict): {
            "time": "13:15",
            "vix": 18,
            "trend": "sideways",
            "volatility": "high",
            "is_expiry": True
        }

    Returns:
        dict: {
            "strategy_id": "expiry_options_sell",
            "strategy_name": "Iron Condor Builder"
        }
    """

    # Basic example logic
    if context.get("is_expiry") and context.get("trend") == "sideways" and context.get("volatility") == "high":
        return {
            "strategy_id": "expiry_options_sell",
            "strategy_name": "Iron Condor Builder"
        }
    
    if context.get("vix", 0) > 20 and context.get("trend") == "down":
        return {
            "strategy_id": "protective_put",
            "strategy_name": "Protective Put Strategy"
        }

    if context.get("vix", 0) < 13 and context.get("trend") == "up":
        return {
            "strategy_id": "bull_call_spread",
            "strategy_name": "Bull Call Spread"
        }

    # Default fallback
    return {
        "strategy_id": "default_hold",
        "strategy_name": "Wait & Watch (No Trade)"
                                               }
