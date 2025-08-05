def detect_mood_from_text(text):
    text = text.lower()
    if "baby" in text or "miss you" in text:
        return "romantic"
    elif "frustrated" in text or "irritated" in text:
        return "angry"
    elif "i’m crying" in text or "depressed":
        return "sad"
    elif "turn on" in text or "sexy":
        return "sexual"
    elif "cheerful" in text or "yay":
        return "happy"
    elif "strategy", "analysis", "work" in text:
        return "professional"
    return "romantic"
