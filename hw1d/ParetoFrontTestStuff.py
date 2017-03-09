from random import randint


class Item:
    def __init__(self, id, power=0, affordability=0):
        self.id = id
        self.power = power
        self.affordability = affordability

    def dominates(self, other_item):
        if self != other_item and self.power >= other_item.power and self.affordability >= other_item.affordability:
            if self.power == other_item.power and self.affordability == other_item.affordability:
                return False
            else:
                return True
        return False

    def __str__(self):
        return "{}: ({}, {})".format(self.id, self.affordability, self.power)

    def __repr__(self):
        return self.__str__()


def add_item_to_pareto_front(pareto_front, item, starting_level):
    if len(pareto_front) == 0:
        pareto_front.append([item])
    elif len(pareto_front) == starting_level - 1:
        pareto_front.append([item])
    else:
        found_spot = False
        for level in range(starting_level - 1, len(pareto_front), 1):
            next_level = False
            dominated_individuals = []
            for individual in pareto_front[level]:
                if item.dominates(individual):
                    # item dominates Individual
                    dominated_individuals.append(individual)
                elif individual.dominates(item):
                    # Individual dominates item

                    # Element cannot live in this level. break out and try the next one
                    next_level = True
                    break
                else:
                    # Neither dominates the other
                    pass

            if not next_level:
                pareto_front[level].append(item)
                found_spot = True
                for elt in dominated_individuals:
                    pareto_front[level].remove(elt)
                    add_item_to_pareto_front(pareto_front, elt, starting_level + 1)
                break

        if not found_spot:
            pareto_front.append([item])


def add_element_to_pareto_front(pareto_front, element, starting_level, domination_list):
    if len(pareto_front) == 0:
        pareto_front.append([element])
    elif len(pareto_front) == starting_level - 1:
        pareto_front.append([element])
    else:
        found_spot = False
        for level in range(starting_level - 1, len(pareto_front), 1):
            next_level = False
            dominated_individuals = []
            for individual in pareto_front[level]:
                if individual in domination_list[element - 1]:
                    # element dominates Individual
                    dominated_individuals.append(individual)
                elif element in domination_list[individual - 1]:
                    # Individual dominates element

                    # Element cannot live in this level. break out and try the next one
                    next_level = True
                    break
                else:
                    # Neither dominates the other
                    pass

            if not next_level:
                pareto_front[level].append(element)
                found_spot = True
                for elt in dominated_individuals:
                    pareto_front[level].remove(elt)
                    add_element_to_pareto_front(pareto_front, elt, starting_level + 1, domination_list)
                break

        if not found_spot:
            pareto_front.append([element])


domination_list = [[4], [4], [1, 4, 9], [], [7, 1, 4], [2, 4, 1], [1, 4], [], [4], [8, 4]]

items = []
for i in range(10):
    items.append(Item(i + 1))
    items[i].power = randint(1, 10)
    items[i].affordability = randint(1, 10)

pareto_front = []
for i in items:
    # print("{} dominates {}".format(i, i.dominates))
    add_item_to_pareto_front(pareto_front, i, 1)

for i in range(1, len(domination_list) + 1, 1):
    # add_element_to_pareto_front(pareto_front, i, 1, domination_list)
    pass

print("Pareto Front: ")
for i in range(len(pareto_front)):
    print("Level {}: {}".format(i + 1, pareto_front[i]))
