"""Question, governing section and required answer pattern."""
CASES = [
    ("Which form requests a relocation advance?", "105.1", r"ADV[-‑– ]4"),
    ("What is the household shipment limit?", "102.2", r"5,?000"),
    ("What is the combined final travel limit?", "102.3", r"1,?200"),
    ("What is the nightly lodging cap?", "102.4", r"150"),
    ("How many nights of temporary lodging are covered?", "102.4", r"\b(ten|10)\b"),
    ("How many calendar days do I have to file under the current edition?",
     "103.3", r"\b(thirty|30)\b"),
    ("What number does the portal issue when I submit my claim?",
     "103.3", r"confirmation"),
    ("Who issues the written move authorization?", "101.2", r"people operations"),
]
