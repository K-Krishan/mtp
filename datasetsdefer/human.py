import numpy as np 
import pandas as pd 

def compute_label_counts(human_predictions, label_mapping):
    """
    Compute label counts for each instance based on multiple human predictions.
    
    Args:
        human_predictions (pd.DataFrame): N x K matrix where each row represents 
                                          multiple human annotations.
        label_mapping (dict): Dictionary mapping possible label values 
                              (e.g., {0: "negative", 1: "positive"}).
    
    Returns:
        pd.DataFrame: DataFrame with added "label_counts" column.
    """
    label_counts = human_predictions.apply(lambda row: row.value_counts().reindex(label_mapping.keys(), fill_value=0).to_list(), axis=1)
    return label_counts


def compute_label_distribution(label_counts):
    """
    Convert label counts into a probability distribution.
    
    Args:
        label_counts (list of lists): List where each entry is a count vector 
                                      for a sample (e.g., [3, 5, 2] for 3x label 0, 5x label 1, 2x label 2).
    
    Returns:
        list of np.array: Probability distributions per instance.
    """
    return [np.array(counts) / sum(counts) for counts in label_counts]


def sample_human_prediction(label_distribution, strategy="random"):
    """
    Sample human predictions based on a given strategy.
    
    Args:
        label_distribution (list of np.array): List of probability distributions per instance.
        strategy (str): Sampling strategy; "random" or "mturk-style".
    
    Returns:
        list: Human predictions per instance.
    """
    sampled_predictions = []
    
    for dist in label_distribution:
        if strategy == "random":
            sampled_predictions.append(np.random.choice(len(dist), p=dist))
        elif strategy == "mturk-style":
            # Simulating the MTurk logic: Choose the most common response with a chance of flipping
            most_common = np.argmax(dist)
            flip = np.random.rand() < 0.2  # Example: 20% chance of flipping
            sampled_predictions.append(most_common if not flip else (1 - most_common))
    
    return sampled_predictions
