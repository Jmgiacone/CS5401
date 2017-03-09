class Individual:
    def __init__(self, position, id, tree, pac_man):
        self.position = position
        self.prev_position = None
        self.id = id
        self.pac_man = pac_man
        self.tree = tree