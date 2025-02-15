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

def synth(y, d, synth=[0.8,0.6]):
    h = []
    for i in range(len(y)):
        if np.random.rand() < synth[d[i]]:
            h.append(y[i])
        else:
            h.append(1-y[i]) # change it to consider label set later
    return np.array(h)
def make_biased_humans(y, d, accuracy=0.8, odds_diff=0.2):
    n1 = (d==0).sum()
    n2 = (d==1).sum()
    n = len(d)

    a1 = (accuracy*n - odds_diff * n2) / n
    a2 = (accuracy*n + odds_diff * n1) / n
    h = synth(y, d, synth=[a1, a2])
    return h
def biased_synth_multiple_demographics(y, d, sensitive_label, accuracy=0.8, odds_diff=0.2):
    # demographic_count = np.max(d)+1 # assuming 0 to d
    # probs = []
    # for i in range(demographic_count):
    #     if i == sensitive_label:
    #         probs.append(accuracy - odds_diff*(demographic_count-1)/demographic_count)
    #     else:
    #         probs.append(accuracy + odds_diff/demographic_count)
    d_sensitive = (~(d == sensitive_label)).astype(int)
    return make_biased_humans(y, d_sensitive, accuracy, odds_diff)

import numpy as np

def synth_multiclass(y, d, synth, num_classes=None):
    """
    Generate synthetic labels for a multiclass problem.
    
    Parameters:
        y : array-like of true labels (assumed to be integer-coded)
        d : array-like demographic group indicator (e.g., 0 or 1)
        synth : list or array with two elements [accuracy_group0, accuracy_group1]
        num_classes : number of classes; if None, inferred as max(y)+1
        
    Returns:
        h : numpy array of synthetic labels
    """
    if num_classes is None:
        num_classes = int(np.max(y)) + 1
        
    h = []
    for i in range(len(y)):
        # Use the appropriate accuracy for the demographic group d[i]
        if np.random.rand() < synth[d[i]]:
            h.append(y[i])
        else:
            # For error, choose a wrong label uniformly at random from the other labels.
            wrong_labels = list(range(num_classes))
            wrong_labels.remove(y[i])
            h.append(np.random.choice(wrong_labels))
    return np.array(h)

def make_biased_humans_multiclass(y, d, accuracy=0.8, odds_diff=0.2):
    """
    Create synthetic human labels for a multiclass classification task,
    adjusting group-specific accuracy based on an overall accuracy and an 
    equalised odds difference fairness measure.
    
    Parameters:
        y : array-like of true labels (integer-coded)
        d : array-like demographic group indicator (assumed binary: 0 or 1)
        accuracy : overall average accuracy desired across groups.
        odds_diff : the difference in accuracy (or odds difference) between the two groups.
    
    Returns:
        h : numpy array of synthetic labels.
    """
    n1 = (d == 0).sum()
    n2 = (d == 1).sum()
    n = len(d)
    
    # Compute per-group accuracies.
    # Note: These formulas balance overall accuracy and fairness (equalised odds difference)
    a1 = (accuracy * n - odds_diff * n2) / n
    a2 = (accuracy * n + odds_diff * n1) / n
    
    # Generate synthetic labels using the multiclass synthesis function.
    h = synth_multiclass(y, d, synth=[a1, a2])
    return h

# Example usage:
# y = np.array([...])        # true labels, e.g., values in {0, 1, 2, ...}
# d = np.array([...])        # demographics, e.g., 0 or 1 for two groups
# synthetic_labels = make_biased_humans_multiclass(y, d, accuracy=0.85, odds_diff=0.15)

def exact_fair_hatespeech(y):
    probs = np.array(
        [[0.4, 0.3, 0.3],
        [0.3, 0.4, 0.3],
        [0.3, 0.3, 0.4]]
    )
    h = []
    for true_label in y:
        h.append(np.random.choice([0,1,2], p=probs[true_label]))
    h = np.array(h)
    return h

