"""Find-S demo: finds the most specific hypothesis consistent with all POSITIVE examples.
Task: 'Is traffic HIGH?'   Attributes: Time, Vehicles, Weather, Day"""
data = [
    # Time,     Vehicles, Weather,  Day,       HighTraffic?
    ("Morning", "High",   "Clear",  "Weekday", "Yes"),
    ("Morning", "High",   "Cloudy", "Weekday", "Yes"),
    ("Morning", "Low",    "Clear",  "Weekday", "No"),
    ("Evening", "High",   "Rain",   "Weekday", "Yes"),
]
h = None
for i, (*x, label) in enumerate(data, 1):
    if label == "Yes":
        h = list(x) if h is None else [a if a == b else "?" for a, b in zip(h, x)]
    print(f"After example {i} ({label}): h = {h}")
print("\nFinal hypothesis:", h, " ('?' means any value is accepted)")
