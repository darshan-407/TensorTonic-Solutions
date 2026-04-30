import numpy as np

def entropy_node(y):
    # Handle empty input
    if len(y) == 0:
        return 0.0
    
    # Get class counts
    _, counts = np.unique(y, return_counts=True)
    
    # Convert counts to probabilities
    probs = counts / counts.sum()
    
    # Filter out zero probabilities
    probs = probs[probs > 0]
    
    # Compute entropy using base-2 logarithm
    entropy = -np.sum(probs * np.log2(probs))
    
    return float(entropy)
