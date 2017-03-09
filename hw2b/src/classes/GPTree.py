from .GPTreeNode import GPTreeNode, NodeType
from copy import deepcopy
from random import choice


class GPTree:
    def __str__(self):
        return self.root_node.__str__()

    @staticmethod
    def subtree_mutation(tree, max_depth):
        random_node = GPTree.get_random_node_in_tree(tree)

        if random_node.node_type is not NodeType.FUNCTION:
            # It's the root node
            tree.root_node = GPTree.grow(max_depth).root_node
        else:
            if choice([0, 1]) == 0:
                random_node.left_child = GPTree.grow(max_depth).root_node
            else:
                random_node.right_child = GPTree.grow(max_depth).root_node

    @staticmethod
    def random_node_in_tree_recursive(node, nodes):
        nodes.append(node)

        if node.node_type is NodeType.FUNCTION:
            GPTree.random_node_in_tree_recursive(node.left_child, nodes)
            GPTree.random_node_in_tree_recursive(node.right_child, nodes)

    @staticmethod
    def get_random_node_in_tree(tree):
        if tree.root_node.node_type is not NodeType.FUNCTION:
            return tree.root_node
        nodes = []
        GPTree.random_node_in_tree_recursive(tree.root_node, nodes)

        selected_node = None

        while selected_node is None:
            selected_node = choice(nodes)

            if selected_node.node_type is not NodeType.FUNCTION:
                selected_node = None
        return selected_node

    @staticmethod
    def sub_tree_crossover(tree1, tree2):
        # Make deep copies because these are two new individuals
        copy1 = deepcopy(tree1)
        copy2 = deepcopy(tree2)

        parent_rand_node1 = GPTree.get_random_node_in_tree(copy1)
        parent_rand_node2 = GPTree.get_random_node_in_tree(copy2)

        # Choose a random child from each parent
        child_choice1, child_choice2 = choice([0, 1]), choice([0, 1])

        # Actual random node is a random child of the parent, that way we can fix the parent pointer
        if parent_rand_node1.node_type is NodeType.FUNCTION:
            rand_node1 = parent_rand_node1.left_child if child_choice1 == 0 else parent_rand_node1.right_child
        else:
            rand_node1 = parent_rand_node1
            
        if parent_rand_node2.node_type is NodeType.FUNCTION:
            rand_node2 = parent_rand_node2.left_child if child_choice2 == 0 else parent_rand_node2.right_child
        else:
            rand_node2 = parent_rand_node2

        # None out the child pointer in the parents, then put the other node in its place
        if parent_rand_node1.node_type is NodeType.FUNCTION:
            if child_choice1 == 0:
                parent_rand_node1.left_child = None
                parent_rand_node1.left_child = rand_node2
            else:
                parent_rand_node1.right_child = None
                parent_rand_node1.right_child = rand_node2
        else:
            copy1.root_node = rand_node2

        if parent_rand_node2.node_type is NodeType.FUNCTION:
            if child_choice2 == 0:
                parent_rand_node2.left_child = None
                parent_rand_node2.left_child = rand_node1
            else:
                parent_rand_node2.right_child = None
                parent_rand_node2.right_child = rand_node1
        else:
            copy2.root_node = rand_node1

        if choice([0, 1]) == 0:
            return copy1
        else:
            return copy2

    @staticmethod
    def print_level_order(tree):
        return GPTreeNode.in_order_traversal(tree.root_node)

    @staticmethod
    def grow(max_depth):
        # Grow method
        tree = GPTree()
        tree.max_depth = max_depth
        tree.num_nodes = GPTreeNode.grow(tree.root_node, max_depth)

        return tree

    @staticmethod
    def full(max_depth):
        # Full method
        tree = GPTree()
        tree.max_depth = max_depth
        tree.num_nodes = GPTreeNode.full(tree.root_node, max_depth)

        return tree

    def __init__(self):
        self.id = None
        self.num_nodes = 1
        self.root_node = GPTreeNode()
        self.max_depth = 0
        self.fitness = None

    def evaluate(self, sensor_values):
        return self.root_node.evaluate(sensor_values)
