import sys


class Genome:
    def __init__(self, id, wall_time=0, memory_usage=0, decisions=0):
        self.objectives = {"wall_time": wall_time, "memory_usage": memory_usage, "decisions": decisions}
        self.id = id

    def dominates(self, other_item):
        if self != other_item and self.objectives["wall_time"] >= other_item.objectives["wall_time"] \
                and self.objectives["memory_usage"] >= other_item.objectives["memory_usage"] \
                and self.objectives["decisions"] >= other_item.objectives["decisions"]:
            if self.objectives["wall_time"] == other_item.objectives["wall_time"] \
                    and self.objectives["memory_usage"] == other_item.objectives["memory_usage"] \
                    and self.objectives["decisions"] == other_item.objectives["decisions"]:
                return False
            else:
                return True
        return False

    def __str__(self):
        return "{}: ({}, {}, {})".format(self.id, self.objectives["wall_time"], self.objectives["memory_usage"],
                                         self.objectives["decisions"])

    def __repr__(self):
        return self.__str__()


def front1_better_than_front2(front1, front2):
    front1_dominates = 0
    front2_dominates = 0

    for genome1 in front1:
        for genome2 in front2:
            if genome1.dominates(genome2):
                front1_dominates += 1
                break

    for genome2 in front2:
        for genome1 in front1:
            if genome2.dominates(genome1):
                front2_dominates += 1
                break

    front1_ratio = front1_dominates / len(front2)
    front2_ratio = front2_dominates / len(front1)

    return front1_ratio > front2_ratio


def parse_file_to_genome_list(file_in):
    x = 0
    genome_list = [[]]
    for line in file_in:
        if line == "\n":
            x += 1
            genome_list.append([])
        elif line[0] != "R":
            line = line.rstrip("\n")
            genome = Genome(1)
            split_list = line.split("\t")

            genome.objectives["wall_time"] = float(split_list[0])
            genome.objectives["memory_usage"] = float(split_list[1])
            genome.objectives["decisions"] = float(split_list[2])

            genome_list[x].append(genome)
    return genome_list


def measure_diversity(front):
    return measure(front, ["wall_time", "memory_usage", "decisions"],
                   {"wall_time": -150, "memory_usage": -100000, "decisions": 0},
                   {"wall_time": 100, "memory_usage": 10000, "decisions": 100000})


def measure(front, objectives, mins, maxs):

    """

    Calculates the normalized hyper-volume between each point on a Pareto front and its neighbors

    Returns the percentage of the total normalized volume NOT taken up by these volumes

        A higher return value corresponds to a better distributed Pareto front

    front: non-empty list of class objects with an objectives dictionary member variable

    objectives: list of objective names (needs to match what's in the individual's objectives dictionary)

    mins: dictionary with objective names as keys and the minimum possible value for that objective as values

    maxs: dictionary with objective names as keys and the maximum possible value for that objective as values

    """

    # This will store the hyper-volume between neighboring individuals on the front; initialize all volumes to 1

    volumes = {individual: 1.0 for individual in front}

    # There is one more volume of interest than there is points on the front, so associate it with the max value

    volumes['max'] = 1.0

    for objective in objectives:

        # Sort the front by this objective's values

        sorted_front = sorted(front, key=lambda x: x.objectives[objective])

        # Calculate the volume between the first solution and minimum

        volumes[sorted_front[0]] *= float(sorted_front[0]
                                          .objectives[objective]-mins[objective]) / (maxs[objective]-mins[objective])

        # Calculate the volume between adjacent solutions on the front

        for i in range(1, len(sorted_front)):

            volumes[sorted_front[i]] *= float(sorted_front[i].objectives[objective]-sorted_front[i-1]
                                              .objectives[objective]) / (maxs[objective]-mins[objective])

        # Calculate the volume between the maximum and the last solution

        volumes['max'] *= float(maxs[objective]-sorted_front[-1]
                                .objectives[objective]) / (maxs[objective]-mins[objective])

    # The normalized volume of the entire objective space is 1.0, subtract the volumes we calculated to turn this into 
    # maximization

    return 1.0 - sum(volumes.values())

if len(sys.argv) != 3:
    print("Error")
    exit(1)

print("Param 1: {}\nParam 2: {}".format(sys.argv[1], sys.argv[2]))
file1 = open(sys.argv[1], "r")
file2 = open(sys.argv[2], "r")

genome_list_1 = parse_file_to_genome_list(file1)
genome_list_2 = parse_file_to_genome_list(file2)

win_ratio_genome_1 = []
win_ratio_genome_2 = []
for run1 in genome_list_1:
    wins_1 = 0
    for run2 in genome_list_2:
        if front1_better_than_front2(run1, run2):
            wins_1 += 1
    win_ratio_genome_1.append(wins_1 / len(genome_list_2))

for run2 in genome_list_2:
    wins_2 = 0
    for run1 in genome_list_1:
        if front1_better_than_front2(run2, run1):
            wins_2 += 1
    win_ratio_genome_2.append(wins_2 / len(genome_list_1))

print("\n{}".format(sys.argv[1]))
for wins1 in win_ratio_genome_1:
    print(wins1)

print("\n{}".format(sys.argv[2]))
for wins2 in win_ratio_genome_2:
    print(wins2)

print("\nFront 1")
for run in genome_list_1:
    print("Diversity: {}".format(measure_diversity(run)))

print("\nFront 2")
for run in genome_list_2:
    print("Diversity: {}".format(measure_diversity(run)))
