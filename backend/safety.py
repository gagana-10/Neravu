URGENT_KEYWORDS = [
    # English
    "chest pain",
    "can't breathe",
    "cannot breathe",
    "difficulty breathing",
    "severe bleeding",
    "unconscious",
    "fainted",
    "stroke",
    "heart attack",
    "suicide",
    "overdose",

    # Hindi
    "सीने में दर्द",
    "सांस नहीं आ रही",
    "सांस लेने में दिक्कत",

    # Kannada
    "ಎದೆ ನೋವು",
    "ಉಸಿರಾಡಲು ಆಗುತ್ತಿಲ್ಲ",
    "ಉಸಿರಾಟದ ತೊಂದರೆ",

    # Telugu
    "ఛాతీ నొప్పి",
    "ఊపిరి ఆడటం లేదు",
    "శ్వాస తీసుకోవడంలో ఇబ్బంది",

    # Tamil
    "நெஞ்சு வலி",
    "மூச்சு விட முடியவில்லை",
    "மூச்சு திணறல்",

    # Marathi
    "छातीत दुखत आहे",
    "श्वास घेता येत नाही",
    "श्वास घेण्यास त्रास"
]


def check_safety(message):
    message_lower = message.lower()

    for keyword in URGENT_KEYWORDS:
        if keyword in message_lower:
            return True

    return False