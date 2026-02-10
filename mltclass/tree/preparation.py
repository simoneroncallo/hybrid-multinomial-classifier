import numpy as np

def get_bisection(classes: list, rng):
    """
    Recursively generate a random binary tree over classes.
    Return a nested tuple structure.
    """
    if len(classes) == 1:
        return classes[0]
    rng.shuffle(classes) # Shuffle before splitting
    # Split nodes
    mid = len(classes) // 2
    left = classes[:mid]
    right = classes[mid:]

    return (get_bisection(left, rng), get_bisection(right, rng))

def get_leaves(subtree):
    """
    Return a list of leaves in the subtree.
    """
    if isinstance(subtree, int):
        return [subtree] # Single leave
    left, right = subtree
    return get_leaves(left) + get_leaves(right)

def get_nodes(tree, max_depth, current_depth=0):
    """
    Return all the binary nodes at a given depth.
    """
    if current_depth == max_depth:
        return [get_leaves(tree)]
    if isinstance(tree, int):
        return [] # Stop at leaves

    left, right = tree
    return (get_nodes(left, max_depth, current_depth+1) + get_nodes(right, max_depth, current_depth+1))

def get_tree(labels: np.ndarray, rng):
    """
    Generate the binary decision tree.
    """
    partition = get_bisection(labels.tolist(), rng)
    depth, tree = 0, []
    # print('Tree')
    while True:
        node = get_nodes(partition, depth)
        if node != []:
            tree.append(node)
            # print(f"Depth {depth}: {node}")
            depth += 1
        elif node == []: # Stops with empty leaves
            break
    return tree, partition, depth

def get_multinomial(x, models, tree, partition, label2idx):
    """
    Recursively evaluate the tree and predict the multinomial label.
    """
    def flatten(x: tuple):
        """
        Recursively flatten a given tuple, e.g. ((5,8),(1,(4,6))) -> (5,8,1,4,6).
        """
        for item in x:
            if isinstance(item, tuple):
                yield from flatten(item)
            else:
                yield item
                
    if isinstance(partition, int):
        return partition # Return leaves
    
    node_labels = tuple(flatten(partition)) # Available labels in the node
    node_idx = label2idx[node_labels] # Flattened index

    left, right = partition
    out = models[node_idx](x).item()
    if out <= 0.5: # Go left
        return get_multinomial(x, models, tree, left, label2idx)
    else: # Go right
        return get_multinomial(x, models, tree, right, label2idx)