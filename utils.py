def detect_mood_from_text(text):
    text = text.lower()
    if "baby" in text or "miss you" in text:
        return "romantic"
    elif "frustrated" in text or "irritated" in text:
        return "angry"
    elif "i’m crying" in text or "depressed" in text:
        return "sad"
    elif "turn on" in text or "sexy" in text:
        return "sexual"
    elif "cheerful" in text or "yay" in text:
        return "happy"
    elif any(word in text for word in ["strategy", "analysis", "work"]):
        return "professional"
    return "romantic"
