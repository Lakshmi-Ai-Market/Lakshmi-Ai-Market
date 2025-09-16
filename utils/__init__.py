from .indicators import TechnicalIndicators

def detect_mood_from_text(text: str) -> str:
    text = text.lower()
    if "happy" in text:
        return "happy"
    elif "sad" in text:
        return "sad"
    elif "angry" in text:
        return "angry"
    else:
        return "neutral"

__all__ = ["TechnicalIndicators", "detect_mood_from_text"]