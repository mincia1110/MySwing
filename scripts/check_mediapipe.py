import mediapipe as mp
print(f"Version: {mp.__version__}")
print(f"Has solutions: {hasattr(mp, 'solutions')}")
attrs = [x for x in dir(mp) if not x.startswith('_')]
print(f"Attributes: {attrs}")
