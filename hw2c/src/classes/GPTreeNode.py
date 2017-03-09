from enum import Enum
from random import uniform, randrange, choice


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        b = .000000000000001
    return a / b


class NodeType(Enum):
    SENSOR = 1
    CONSTANT = 2
    FUNCTION = 3


class SensorType(Enum):
    NEAREST_ENEMY = 1
    NEAREST_PILL = 2
    ADJACENT_WALLS = 3
    NEAREST_FRUIT = 4
    NEAREST_ALLY = 5


class FunctionType(Enum):
    ADD = 1
    SUBTRACT = 2
    MULTIPLY = 3
    DIVIDE = 4
    RAND = 5


class GPTreeNode:
    def __str__(self, depth=0):
        ret = ""

        # Print own value
        if self.node_type is not NodeType.FUNCTION:
            # Print value and be done
            if self.node_type == NodeType.CONSTANT:
                ret += "\n" + ("    " * depth) + str(self.data)
            else:
                ret += "\n" + ("    " * depth) + str(self.sensor_type.name)
        else:
            # Recurse
            ret += self.right_child.__str__(depth + 1)
            ret += "\n" + ("    " * depth) + str(self.data.name)
            ret += self.left_child.__str__(depth + 1)

        return ret

    @staticmethod
    def in_order_traversal(node):
        output = ""
        nodes = []
        nodes_in_level = 1
        nodes_in_next = 0
        nodes.append(node)

        while len(nodes) != 0:
            curr_node = nodes[0]
            nodes.pop(0)
            nodes_in_level -= 1

            if curr_node is not None:
                if curr_node.node_type is NodeType.FUNCTION:
                    output += curr_node.data.name + " "
                elif curr_node.node_type is NodeType.CONSTANT:
                    output += str(curr_node.data) + " "
                else:
                    output += curr_node.sensor_type.name + " "
                nodes.append(curr_node.left_child)
                nodes.append(curr_node.right_child)
                nodes_in_next += 2

            if nodes_in_level == 0:
                output += "\n"
                nodes_in_level = nodes_in_next
                nodes_in_next = 0

        return output

    @staticmethod
    def grow(node, max_depth):
        # If we have enough levels to make a function node
        if max_depth > 1:
            # Set the node type to a random type in {SENSOR, CONSTANT, FUNCTION}
            node.node_type = choice(list(NodeType))
        else:
            node.node_type = [NodeType.CONSTANT, NodeType.SENSOR][randrange(2)]
        
        if node.node_type is NodeType.SENSOR:
            # Set sensor type to a random sensor type
            node.sensor_type = choice(list(SensorType))
            return 1
        elif node.node_type is NodeType.CONSTANT:
            # Set constant to a random value
            node.data = uniform(0, 9)
            return 1
        elif node.node_type is NodeType.FUNCTION:
            node.data = choice(list(FunctionType))

            # Initialize children
            node.left_child = GPTreeNode()
            node.right_child = GPTreeNode()

            # Grow children
            nodes_l = GPTreeNode.grow(node.left_child, max_depth - 1)
            nodes_r = GPTreeNode.grow(node.right_child, max_depth - 1)

            return nodes_l + nodes_r + 1
        else:
            print("Error!")
            exit(1)

    @staticmethod
    def full(node, max_depth):
        if max_depth > 1:
            # Only put non-terminal nodes
            node.node_type = NodeType.FUNCTION

            # Give the node a function
            node.data = choice(list(FunctionType))

            # Initialize children
            node.left_child = GPTreeNode()
            node.right_child = GPTreeNode()

            # Grow children
            l_num_nodes = GPTreeNode.full(node.left_child, max_depth - 1)
            r_num_nodes = GPTreeNode.full(node.right_child, max_depth - 1)

            return l_num_nodes + r_num_nodes + 1
        else:
            # Throw down a terminal node
            node.node_type = [NodeType.CONSTANT, NodeType.SENSOR][randrange(2)]

            if node.node_type is NodeType.SENSOR:
                # Set sensor type to a random sensor type
                node.sensor_type = choice(list(SensorType))
                return 1
            elif node.node_type is NodeType.CONSTANT:
                # Set constant to a random value
                node.data = uniform(-10, 10)
                return 1

    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.data = None
        self.node_type = None
        self.sensor_type = None

    def evaluate(self, sensor_values):
        if self.node_type is NodeType.SENSOR:
            return sensor_values[self.sensor_type]
        elif self.node_type is NodeType.CONSTANT:
            return self.data
        elif self.node_type is NodeType.FUNCTION:
            left_value = self.left_child.evaluate(sensor_values)
            right_value = self.right_child.evaluate(sensor_values)

            if self.data is FunctionType.ADD:
                return add(left_value, right_value)
            elif self.data is FunctionType.SUBTRACT:
                return subtract(left_value, right_value)
            elif self.data is FunctionType.MULTIPLY:
                return multiply(left_value, right_value)
            elif self.data is FunctionType.DIVIDE:
                return divide(left_value, right_value)
            elif self.data is FunctionType.RAND:
                # Special rand case, make sure args are in order
                return uniform(min(left_value, right_value), max(left_value, right_value))
        else:
            print("Error!")
            exit(1)
