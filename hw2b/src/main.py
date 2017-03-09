from classes.GPTree import GPTree
from classes.GPTreeNode import SensorType
from random import randrange, randint, seed, choice, uniform
import sys
import configparser
from os.path import exists
from os import makedirs
from classes.Individual import Individual
from classes.ParentSelectionMethod import ParentSelectionMethod
from classes.SurvivalSelectionMethod import SurvivalSelectionMethod
from math import floor, ceil
from copy import deepcopy


def manhattan_distance(point1, point2):
    if point1 is None or point2 is None:
        return 0
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


def closest_symbol(board, start, symbol):
    if board[start[0]][start[1]] == symbol:
        return start

    queue = []
    where_from = {}
    queue.append(start)

    while len(queue) != 0:
        node = queue.pop(0)

        where_from[node] = True

        for dir in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = node[0] + dir[0], node[1] + dir[1]

            if neighbor not in where_from:
                where_from[neighbor] = True
                try:
                    if neighbor[0] >= 0 and neighbor[1] >= 0:
                        if board[neighbor[0]][neighbor[1]] == symbol:
                            # Neighbor is in bounds and we found a pellet
                            return neighbor
                        else:
                            queue.append(neighbor)

                except IndexError:
                    # Out of bounds
                    pass
    return None


def distance_to_closest_symbol(board, start, symbol):
    return manhattan_distance(start, closest_symbol(board, start, symbol))


def main():
    # Exactly 1 CLI argument required
    if len(sys.argv) != 2:
        print("Usage: ./{} <config>".format(sys.argv[0]))
        exit(1)

    config = configparser.ConfigParser()
    config_file = sys.argv[1]

    try:
        config.read_file(open(config_file))
    except FileNotFoundError:
        print("Error: Config File \"{}\" not found.".format(sys.argv[1]))
        exit(1)

    # Grab pertinent values from config
    num_runs = int(config["General Config"]["Runs Per Experiment"])
    game_seed = config["General Config"]["Seed"]

    # Test the seed variable. If it's an int, then seed with it, otherwise seed with a random seed
    try:
        game_seed = int(game_seed)
        seed(game_seed)
    except ValueError:
        # It's not an integer, so assume they want a time-initialized seed
        game_seed = randint(0, sys.maxsize)
        seed(game_seed)

    # Tuple with the score of the game and a string signifying the world file
    best_game = 0, None

    # Open log file
    log_file = open(str(config["Logging Config"]["Log File"]), "w+", 1)
    log_file.write("Result Log\n")

    config_string = ""
    # Aggregate config to a string
    for section in config.sections():
        for (key, value) in config.items(section):
            if key != "seed":
                config_string += "{} = {}\n".format(key, value)
            else:
                config_string += "seed = {}\n".format(game_seed)

    # Print config to the console
    print("=== Config ===")
    print(config_string)
    print("=== End Config ===")

    # Print config in the log
    log_file.write("=== Config ===\n")
    log_file.write(config_string)
    log_file.write("=== End Config ===\n")

    # Grab config file name
    period_index = config_file.find(".")
    forward_slash_index = config_file.find("/", 0, -1)
    config_file = config_file[forward_slash_index + 1:period_index]
    world_file_path = "games/" + config_file + "/" + str(game_seed)

    # Check to see if a folder exists for this config file
    if not exists(world_file_path):
        makedirs(world_file_path)

    best_score = -1
    best_world = ""
    best_member = None
    for run in range(num_runs):
        print("Run {}".format(run + 1))
        log_file.write("Run {}\n".format(run + 1))

        if not exists(world_file_path + "/" + "run" + str(run + 1)):
            makedirs(world_file_path + "/" + "run" + str(run + 1))

        run_best_score, run_best_world, run_best_member = single_run(config, world_file_path + "/" + "run" + str(run + 1), log_file)
        log_file.write("\n")

        # If this run had a best game that is better than the current best
        if run_best_score > best_score:
            best_score = run_best_score
            best_world = run_best_world

        if best_member is None:
            best_member = run_best_member
        elif run_best_member.fitness > best_member.fitness:
            best_member = run_best_member

    solution_file = open(str(config["Logging Config"]["Solution File"]), "w+")
    solution_file.write(GPTree.print_level_order(best_member))
    best_world_file = open(str(config["Logging Config"]["Best World File"]), "w+")
    best_world_file.write(best_world)

    best_world_file.close()
    solution_file.close()
    log_file.close()


def find_empty_space(board, empty_space_chars):
    """
    :rtype: tuple(int, int)
    """
    rows, columns = len(board), len(board[0])

    space = randrange(rows), randrange(columns)

    if board[space[0]][space[1]] not in empty_space_chars:
        return None

    return space


def get_frontier(board, cell, frontier_list):
    frontier = set()
    r = cell[0]
    c = cell[1]

    for space in frontier_list:
        new_r = r + space[0]
        new_c = c + space[1]

        if new_r >= 0 and new_c >= 0:
            try:
                if board[new_r][new_c] == "#":
                    frontier.add((new_r, new_c))
            except IndexError:
                pass

    return frontier


def prims_maze_generator(board, wall_density):
    """http://stackoverflow.com/questions/29739751/implementing-a-randomly-generated-maze-using-prims-algorithm"""
    frontier_spaces = [(0, 2), (0, -2), (2, 0), (-2, 0)]
    neighbor_spaces = [((0, 2), (0, 1)), ((0, -2), (0, -1)), ((2, 0), (1, 0)), ((-2, 0), (-1, 0))]

    rows, columns = len(board), len(board[0])
    current_walls = rows * columns
    expected_walls = max(1, int(wall_density * (current_walls - 2)))

    # Choose a random cell to start in
    random_cell = randrange(rows), randrange(columns)

    # Set its state to not wall
    board[random_cell[0]][random_cell[1]] = "."
    current_walls -= 1

    # Calculate frontier cells (any blocked cell a distance of 2 away
    frontier_cells = get_frontier(board, random_cell, frontier_spaces)

    # While there are still frontier cells to be seen
    frontier_cells_len = len(frontier_cells)

    while current_walls > expected_walls and frontier_cells_len != 0:
        # Choose a random frontier cell
        random_cell = frontier_cells.pop()

        # Set to a space
        board[random_cell[0]][random_cell[1]] = "."
        current_walls -= 1

        # Find a random neighbor (cell 2 spaces away that isn't a wall)
        space_list = get_neighbors(board, random_cell, neighbor_spaces)

        if len(space_list) != 0:
            # random_space = space_list[randrange(len(space_list))]
            random_space = space_list.pop()
            board[random_space[0]][random_space[1]] = "."
            current_walls -= 1
            space_list.clear()

        # Add frontier cells of random_cell to the list of frontier cells
        frontier_cells |= get_frontier(board, random_cell, frontier_spaces)

        # Re-calculate length
        frontier_cells_len = len(frontier_cells)

    # Need to remove more walls
    possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while current_walls > expected_walls:
        # Find a random wall
        random_cell = find_empty_space(board, ["#"])

        if random_cell is not None:
            for cell in possible_moves:
                r = random_cell[0] + cell[0]
                c = random_cell[1] + cell[1]

                try:
                    # If at least one direct neighbor is an empty space
                    if r >= 0 and c >= 0 and board[r][c] == ".":
                        break
                except IndexError:
                    pass
            else:  # This gets called if the above statement doesn't issue a break
                continue  # This is a wall surrounded by 4 other walls, don't remove it

            board[random_cell[0]][random_cell[1]] = "."
            current_walls -= 1

    return current_walls


def get_neighbors(board, cell, space_list):
    neighbors = set()
    r = cell[0]
    c = cell[1]

    for space in space_list:
        two_away = space[0]
        one_away = space[1]

        new_r = r + two_away[0]
        new_c = c + two_away[1]

        if new_r >= 0 and new_c >= 0:
            try:
                if board[new_r][new_c] == "." and board[r + one_away[0]][c + one_away[1]] == "#":
                    neighbors.add((r + one_away[0], c + one_away[1]))
            except IndexError:
                pass

    return neighbors


def build_initial_world_file(board, file, initial_time, pac_men_positions, ghost_positions):
    file.write(str(len(board[0])) + "\n")
    file.write(str(len(board)) + "\n")

    for pac in pac_men_positions:
        file.write("{} {} {}\n".format("m" + str(pac.id), pac.position[1], pac.position[0]))

    for ghost in ghost_positions:
        file.write("{} {} {}\n".format(ghost.id, ghost.position[1], ghost.position[0]))

    for row in range(len(board)):
        for column in range(len(board[row])):
            if board[row][column] == "#":
                file.write("w {} {}\n".format(column, row))
            elif board[row][column][0] == "m":
                # file.write("{} {} {}\n".format(board[row][column], column, row))
                pass
            elif board[row][column] == "G":
                # file.write("1 {} {}\n".format(column, row))
                # file.write("2 {} {}\n".format(column, row))
                # file.write("3 {} {}\n".format(column, row))
                pass
            elif board[row][column] == "*":
                file.write("p {} {}\n".format(column, row))

    file.write("t {} 0\n".format(initial_time))


def distance_to_closest_point_in_list(position, point_list):
    min_distance = sys.maxsize

    for point in point_list:
        min_distance = min(manhattan_distance(position, point), min_distance)

    return min_distance if min_distance != sys.maxsize else 0


def evaluate_gp_tree(board, individual, ghosts, pac_list, fruit_position):
    pac_man_list = set(pac_list)
    ghost_list = set(ghosts)

    valid_moves = []
    if individual.pac_man:
        # This is the controller for a pac-man

        # Remove this individual from the list, so we don't find closest path of zero
        pac_man_list = pac_man_list - {individual}
        
        # Convert back to a list of tuples
        tmp = []
        for pac_man in pac_man_list:
            tmp.append(pac_man.position)

        pac_man_list = tmp

        tmp = []
        for ghost in ghost_list:
            tmp.append(ghost.position)
        
        ghost_list = tmp
        
        possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
        enemy_list, ally_list = ghost_list, pac_man_list

    else:
        # This is the controller for a ghost

        # Remove this individual from the list, so we don't find closest path of zero
        ghost_list = ghost_list - {individual}
        
        # Convert back to a list of tuples
        tmp = []
        for pac_man in pac_man_list:
            tmp.append(pac_man.position)

        pac_man_list = tmp

        tmp = []
        for ghost in ghost_list:
            tmp.append(ghost.position)

        ghost_list = tmp
        
        possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        enemy_list, ally_list = pac_man_list, ghost_list

    adjacent_walls = 0
    # Generate valid moves
    for move in possible_moves:
        new_pos = individual.position[0] + move[0], individual.position[1] + move[1]

        if new_pos[0] >= 0 and new_pos[1] >= 0:
            try:
                if board[new_pos[0]][new_pos[1]] != "#":
                    valid_moves.append(new_pos)
                else:
                    adjacent_walls += 1
            except IndexError:
                # Out of bonds, consider it a wall
                adjacent_walls += 1
        else:
            # Out of bounds the other way, consider it a wall
            adjacent_walls += 1

    if not individual.pac_man:
        # Assignment 2b has random ghost movement
        return choice(valid_moves)

    max_score = None

    move_to = None
    for move in valid_moves:
        end_pos = move
        # Calculate sensor values
        sensor_values = {SensorType.NEAREST_ENEMY: distance_to_closest_point_in_list(end_pos, enemy_list),
                         SensorType.NEAREST_PILL: distance_to_closest_symbol(board, end_pos, "*"),
                         SensorType.ADJACENT_WALLS: adjacent_walls,
                         SensorType.NEAREST_FRUIT: manhattan_distance(end_pos, fruit_position),
                         SensorType.NEAREST_ALLY: distance_to_closest_point_in_list(end_pos, ally_list)}
        score = individual.tree.evaluate(sensor_values)

        if max_score is None:
            # First iteration, this move auto-wins
            max_score = score
            move_to = end_pos
        elif score > max_score:
            max_score = score
            move_to = end_pos

    # If this line returns a None, there is a problem with checking valid moves
    return move_to


def play_game(rows, columns, wall_density, pill_density, fruit_chance, fruit_score, time_multiplier, world_file,
              num_pac_men, num_ghosts, gp_trees):
    """BONUS 1&2: This whole function was written with an arbitrary number of pac-men and ghosts in mind. Everything is
    generalized down to the list level for that very purpose. Even the controllers are in their own list, so that
    I can pass in all the same one, or a different one for each index"""

    # Initialize book-keeping info
    pac_men = []
    ghosts = []
    fruit_position = None
    game_over = False
    pills_collected = 0
    fruit_collected = 0
    score = 0
    total_time = time_remaining = rows * columns * time_multiplier
    fruit_written = False
    num_pills = 0
    to_remove = set()
    file_contents = ""

    # Open world file for writing
    file = open(world_file, "w+")

    board = ["#"] * rows
    for b in range(rows):
        board[b] = ["#"] * columns

    walls = prims_maze_generator(board, wall_density)

    start = closest_symbol(board, (0, 0), ".")

    board[start[0]][start[1]] = "m"

    for i in range(num_pac_men):
        # BONUS 2: Each pac-man gets their own GPTree
        pac_men.append(Individual(start, i+1, gp_trees[i], True))

    start = closest_symbol(board, (rows - 1, columns - 1), ".")

    for i in range(num_ghosts):
        # Ghosts don't get a GPTree
        ghosts.append(Individual(start, i+1, None, False))

    expected_pills = max(1, int(pill_density * (rows * columns - 1 - walls)))

    # While there are open spaces, place down pills
    while num_pills < expected_pills:
        random_cell = find_empty_space(board, ["."])

        if random_cell is None:
            continue

        board[random_cell[0]][random_cell[1]] = "*"
        num_pills += 1

    file.write(str(columns) + "\n")
    file.write(str(rows) + "\n")
    file_contents += "{}\n{}\n".format(columns, rows)

    for pac in pac_men:
        file.write("{} {} {}\n".format("m" + str(pac.id), pac.position[1], pac.position[0]))
        file_contents += "{} {} {}\n".format("m" + str(pac.id), pac.position[1], pac.position[0])

    for ghost in ghosts:
        file.write("{} {} {}\n".format(ghost.id, ghost.position[1], ghost.position[0]))
        file_contents += "{} {} {}\n".format(ghost.id, ghost.position[1], ghost.position[0])

    for row in range(len(board)):
        for column in range(len(board[row])):
            if board[row][column] == "#":
                file.write("w {} {}\n".format(column, row))
                file_contents += "w {} {}\n".format(column, row)
            elif board[row][column] == "*":
                file.write("p {} {}\n".format(column, row))
                file_contents += "p {} {}\n".format(column, row)

    file.write("t {} 0\n".format(total_time))
    file_contents += "t {} 0\n".format(total_time)

    while not game_over:
        # Attempt to spawn fruit
        if fruit_position is None and randrange(100) + 1 < fruit_chance:
            # Spawn a fruit in the maze
            # Fruit is allowed to spawn on top of ghosts
            
            # Give it 10 tries
            for i in range(10):
                fruit_position = find_empty_space(board, ["."])
                
                if fruit_position is not None:
                    for pac_man in pac_men:
                        if fruit_position == pac_man.position:
                            fruit_position = None
                            break
                    else:
                        break

            if fruit_position is not None:
                fruit_written = False
                board[fruit_position[0]][fruit_position[1]] = "f"
        
        # Base it on length of position vector so that dead pac-men can be removed
        for pac_man in pac_men:
            pac_man.prev_position = pac_man.position
            board[pac_man.position[0]][pac_man.position[1]] = "."
            pac_man.position = evaluate_gp_tree(board, pac_man, ghosts, pac_men, fruit_position)
            # board[pac_man.position[0]][pac_man.position[1]] = "m" + str(pac_man.id)
        
        for ghost in ghosts:
            ghost.prev_position = ghost.position
            ghost.position = evaluate_gp_tree(board, ghost, ghosts, pac_men, fruit_position)

        # Check pac-man collisions
        for pac_man in pac_men:
            # Write this pac-man's position
            file.write("m{} {} {}\n".format(pac_man.id, pac_man.position[1], pac_man.position[0]))
            file_contents += "m{} {} {}\n".format(pac_man.id, pac_man.position[1], pac_man.position[0])

            # Check to see if pac-man was killed by a ghost
            for ghost in ghosts:
                if pac_man.position == ghost.position:
                    # Pac-man shares a spot with a ghost, this one is dead
                    to_remove.add(pac_man)
                    break
                elif pac_man.position == ghost.prev_position and pac_man.prev_position == ghost.position:
                        # He dead
                        to_remove.add(pac_man)
                        break

        # Remove fallen comrades
        for dead_pac in to_remove:
            pac_men.remove(dead_pac)
        to_remove.clear()

        # If there are no more pac-men in the list, game over
        game_over = len(pac_men) == 0

        if not game_over:
            for pac_man in pac_men:
                # Check pills
                if board[pac_man.position[0]][pac_man.position[1]] == "*":
                    pills_collected += 1
                    score = pills_collected * 100 // num_pills + fruit_collected * fruit_score
                    
                # Check fruit
                if pac_man.position == fruit_position:
                    fruit_collected += 1
                    score += fruit_score
                    fruit_position = None
                
                # Update position
                board[pac_man.prev_position[0]][pac_man.prev_position[1]] = "."
                board[pac_man.position[0]][pac_man.position[1]] = "m" + str(pac_man.id)

            time_remaining -= 1

            # Check game-over conditions
            if pills_collected == num_pills:
                # All pills have been collected, game over

                # Score = 100 (all pills collected, so 100%) + (percentage of time left * 100 truncated to an integer +
                #         score obtained from fruit
                score = 100 + ((100 * time_remaining) // (rows * columns * time_multiplier)) + fruit_score * \
                                                                                               fruit_collected
                game_over = True
            elif time_remaining == 0:
                # Time has run out
                game_over = True

        if fruit_position is not None and not fruit_written:
            file.write("f {} {}\n".format(fruit_position[1], fruit_position[0]))
            file_contents += "f {} {}\n".format(fruit_position[1], fruit_position[0])
            fruit_written = True

        for ghost in ghosts:
            file.write("{} {} {}\n".format(ghost.id, ghost.position[1], ghost.position[0]))
            file_contents += "{} {} {}\n".format(ghost.id, ghost.position[1], ghost.position[0])

        file.write("t {} {}\n".format(time_remaining, score))
        file_contents += "t {} {}\n".format(time_remaining, score)

    file.close()
    return score, file_contents


def ramped_half_and_half(population_size, max_depth):
    # Create blank population
    population = [None] * population_size

    for i in range(population_size):
        # Flip a coin for grow or full
        if randrange(2) == 0:
            # print("Grow")
            population[i] = GPTree.grow(max_depth)
        else:
            # print("Full")
            population[i] = GPTree.full(max_depth)
        population[i].id = i + 1
        # print("Individual {}\n{}".format(i + 1, population[i]))
        # GPTree.print_level_order(population[i])

    return population


def evaluate_population_fitness(num_evals, population, num_pac_men, rows, columns, wall_density, pill_density, fruit_chance,
                                fruit_score, time_multiplier, file_path, num_ghosts, parsimony_coefficient, same_controller_for_each):
    """BONUS 1: This function was also written with the bonuses in mind. Based on the number of simultaneou pac-men, this
    function will partition the controllers up and pass them to the game-playing function"""
    population_size = len(population)
    leftovers = population_size % num_pac_men
    counter = 0
    trees = [None] * num_pac_men
    current_max_score = -1
    best_game = ""

    if same_controller_for_each:
        loop_range = population_size
        leftovers = 0
    else:
        # BONUS 2: Breaks pac-men into group so multiple controllers can be used per round
        loop_range = population_size // num_pac_men
        leftovers = population_size % num_pac_men

    # Purposeful integer division
    for i in range(loop_range):
        for j in range(num_pac_men):
            if same_controller_for_each:
                trees[j] = population[i]
            else:
                trees[j] = population[counter + j]

        num_evals += 1
        # Play the game
        score, game = play_game(rows, columns, wall_density, pill_density, fruit_chance, fruit_score,
                                time_multiplier, file_path + "/" + str(num_evals) + ".world", num_pac_men, num_ghosts,
                                trees)

        for tree in trees:
            tree.fitness = score - parsimony_coefficient * tree.num_nodes

        # Update best game overall
        if score > current_max_score:
            current_max_score = score
            best_game = game
        counter += num_pac_men

    if leftovers != 0:
        for i in range(leftovers):
            trees[i] = population[population_size - 1 - i]

        # Play the game
        score, game = play_game(rows, columns, wall_density, pill_density, fruit_chance, fruit_score,
                                  time_multiplier, file_path + "/" + str(num_evals + 1) + ".world", num_pac_men, num_ghosts,
                                  trees)

        for tree in trees:
            tree.fitness = score - parsimony_coefficient * tree.num_nodes

            # Update best game overall
        if score > current_max_score:
            current_max_score = score
            best_game = game

    return current_max_score, best_game


def min_fitness(population):
    minimum_fitness = None

    for member in population:
        if minimum_fitness is None:
            minimum_fitness = member.fitness
        elif member.fitness < minimum_fitness:
            minimum_fitness = member.fitness

    return minimum_fitness


def fitness_proportional_selection(population, num_parents):
    parents = [None] * num_parents
    a_array = [0.0] * len(population)
    i_value = 0
    current_member = 0
    total_fitness = 0
    running_probability_sum = 0

    # Find the fitness of the worst population member
    minimum_fitness = min_fitness(population)

    for member in population:
        total_fitness += member.fitness - minimum_fitness

    for i in range(len(population)):
        member = population[i]
        if total_fitness == 0:
            member.selection_chance = 1
        else:
            member.selection_chance = (member.fitness - minimum_fitness) / total_fitness
        running_probability_sum += member.selection_chance

        a_array[i] = running_probability_sum

    # Implementation of the Stochastic Universal Sampling Algorithm for FPS
    r_value = uniform(0, 1) * (1 / num_parents)
    while current_member < num_parents and i_value < len(population):
        while r_value <= a_array[i_value]:
            parents[current_member] = population[i_value]
            r_value += (1 / num_parents)
            current_member += 1
        i_value += 1

    for member in population:
        del member.selection_chance

    return parents


def parent_selection(population, num_parents, parent_selection_method, overselection_i_value, overselection_j_value):
    if parent_selection_method is ParentSelectionMethod.FITNESS_PROPORTIONAL:
        return fitness_proportional_selection(population, num_parents)
    elif parent_selection_method is ParentSelectionMethod.OVER_SELECTION:
        # Sort population for easy grabbing of members
        population.sort(key=lambda individual: individual.fitness, reverse=True)

        # Take top i%
        num_elites = ceil(overselection_i_value * len(population))
        elites = population[:num_elites]
        rest = population[num_elites:]

        elite_parents = fitness_proportional_selection(elites, ceil(overselection_j_value * num_parents))
        regular_parents = fitness_proportional_selection(rest, floor((1 - overselection_j_value) * num_parents))

        elite_parents.extend(regular_parents)

        disparity = num_parents - len(elite_parents)

        if disparity != 0:
            # Rounding error killed us, just copy the first element however many times
            for i in range(disparity):
                elite_parents.append(elite_parents[0])

        return elite_parents
    else:
        print("Error: '{}' is not a valid parent selection method".format(parent_selection_method))
        exit(1)


def find_average_fitness_and_best_member(population):
    total_fitness = 0
    best_member = None

    for member in population:
        current = member.fitness
        total_fitness += current

        if best_member is None:
            best_member = member
        elif current > best_member.fitness:
            best_member = member

    return total_fitness / len(population), best_member


def single_run(config, file_path, log_file):
    evals_per_run = int(config["General Config"]['Fitness Evals Per Run'])
    num_pac_men, num_ghosts = int(config["World Config"]["Number of Pac-men"]), \
                              int(config["World Config"]["Number of Ghosts"])
    rows, columns = int(config["World Config"]["World Height"]), int(config["World Config"]["World Width"])
    wall_density = float(config["World Config"]["Wall Density"]) / 100
    pill_density = float(config["World Config"]["Pill Density"]) / 100
    fruit_chance = 100 * float(config["World Config"]["Fruit Spawning Probability"])
    fruit_score = int(config["World Config"]["Fruit Score"])
    time_multiplier = int(config["World Config"]["Time Multiplier"])
    population_size = int(config["EA Config"]["Population Size"])
    offspring_per_generation = int(config["EA Config"]["Offspring per Generation"])
    max_depth = int(config["EA Config"]["Max Depth"])
    mutation_chance = float(config["EA Config"]["Mutation Chance"])
    parent_selection_method = ParentSelectionMethod(int(config["EA Config"]["Parent Selection Method"]))
    survival_selection_method = SurvivalSelectionMethod(int(config["EA Config"]["Survival Selection Method"]))
    tournament_selection_size = int(config["EA Config"]["Tournament Selection Size"])
    parsimony_coefficient = float(config["EA Config"]["Parsimony Penalty Coefficient"])
    terminate_on_num_evals = config["EA Config"]["Terminate on Number of Evals"] == "true"
    terminate_on_stagnant_best_fitness = config["EA Config"]["Terminate on n generations of stagnant best fitness"] \
                                         == "true"
    same_controller_for_each = config["EA Config"]["Multiple Pac-men use one controller"] == "true"
    stagnant_generations_until_termination = int(config["EA Config"]["Stagnant Generations until termination"])
    overselection_i_value = float(config["EA Config"]["i-value"])
    overselection_j_value = float(config["EA Config"]["j-value"])

    current_best_game_score = -1
    current_best_world_file = ""
    num_evals = 0
    stagnant_generations = 0
    total_individuals = 0

    print("Initialization")
    population = ramped_half_and_half(population_size, max_depth)
    print("End Initialization")
    total_individuals += population_size

    fitness, game = evaluate_population_fitness(num_evals, population, num_pac_men, rows, columns, wall_density,
                                                pill_density, fruit_chance, fruit_score, time_multiplier, file_path,
                                                num_ghosts, parsimony_coefficient, same_controller_for_each)
    # Check for best game
    if fitness > current_best_game_score:
        current_best_game_score = fitness
        current_best_world_file = game

    # Increase evals
    num_evals += population_size if same_controller_for_each \
        else population_size // num_pac_men + (1 if population_size % num_pac_men != 0 else 0)

    average_fitness, best_member = find_average_fitness_and_best_member(population)

    generation = 1
    log_file.write("{}\t{}\t{}\n".format(num_evals, average_fitness, best_member.fitness))
    print("{}\t{}\t{}".format(num_evals, average_fitness, best_member.fitness))
    while not terminate(num_evals, evals_per_run, stagnant_generations, stagnant_generations_until_termination,
                        terminate_on_num_evals, terminate_on_stagnant_best_fitness):

        print("Generation {} ({})".format(generation, num_evals))
        previous_best_member = best_member

        print("Parent Selection")
        parents = parent_selection(population, 2 * offspring_per_generation, parent_selection_method,
                                   overselection_i_value, overselection_j_value)

        children = []
        for i in range(0, 2 * offspring_per_generation, 2):
            if uniform(0, 1) < mutation_chance:
                # Perform mutation on a clone of either parent
                child = deepcopy(parents[i + randrange(2)])
                child.fitness = None
                print("Mutate individual {}".format(child.id))

                GPTree.subtree_mutation(child, child.max_depth)
                total_individuals += 1
                child.id = total_individuals
                children.append(child)
            else:
                # Subtree crossover
                print("Crossover individuals {} and {}".format(parents[i].id, parents[i + 1].id))
                child = GPTree.sub_tree_crossover(parents[i], parents[i + 1])
                total_individuals += 1
                child.id = total_individuals
                children.append(child)
            # GPTree.print_level_order(child)

        fitness, game = evaluate_population_fitness(num_evals, children, num_pac_men, rows, columns, wall_density, pill_density,
                                                    fruit_chance, fruit_score, time_multiplier, file_path, num_ghosts,
                                                    parsimony_coefficient, same_controller_for_each)

        num_evals += offspring_per_generation if same_controller_for_each \
            else offspring_per_generation // num_pac_men + (1 if population_size % num_pac_men != 0 else 0)

        # Check for best game
        if fitness > current_best_game_score:
            current_best_game_score = fitness
            current_best_world_file = game

        population = survival_selection(population, children, len(population), tournament_selection_size,
                                        survival_selection_method)


        average_fitness, best_member = find_average_fitness_and_best_member(population)

        if previous_best_member.fitness == best_member.fitness:
            stagnant_generations += 1
            print("Best fitness has been stagnant for {} generations!".format(stagnant_generations))
        else:
            stagnant_generations = 0

        generation += 1

        log_file.write("{}\t{}\t{}\n".format(num_evals, average_fitness, best_member.fitness))
        print("{}\t{}\t{}".format(num_evals, average_fitness, best_member.fitness))

    return current_best_game_score, current_best_world_file, best_member


def survival_selection(population, offspring, num_survivors, tournament_size, survival_selection_method):
    total_population = deepcopy(offspring)
    total_population.extend(population)
    if survival_selection_method is SurvivalSelectionMethod.TRUNCATION:
        total_population.sort(key=lambda t: t.fitness, reverse=True)

        return total_population[:num_survivors]
    elif survival_selection_method is SurvivalSelectionMethod.K_TOURNAMENT_SELECTION:
        survivors_found = 0
        survivors = [None] * num_survivors

        # Create a new field to represent whether or not an individual is entered or not
        for member in total_population:
            member.entered_in_tournament = False

        while survivors_found != num_survivors:
            tournament = [None] * tournament_size
            tournament_members_found = 0

            while tournament_members_found != tournament_size:
                attendee = choice(total_population)

                if not attendee.entered_in_tournament:
                    tournament[tournament_members_found] = attendee
                    attendee.entered_in_tournament = True
                    tournament_members_found += 1

            highest_fitness = None

            for entrant in tournament:
                entrant.entered_in_tournament = False
                if highest_fitness is None:
                    tournament_winner = entrant
                    highest_fitness = entrant.fitness
                elif entrant.fitness > highest_fitness:
                    tournament_winner = entrant
                    highest_fitness = entrant.fitness

            tournament_winner.entered_in_tournament = True
            survivors[survivors_found] = tournament_winner
            survivors_found += 1

        for survivor in survivors:
            del survivor.entered_in_tournament

        return survivors
    else:
        print("Error: '{}' is not a valid survival selection method".format(survival_selection_method))


def terminate(num_evals, evals_per_run, stagnant_generations, stagnant_generations_until_termination,
              terminate_on_num_evals, terminate_on_stagnant_best_fitness):
    if terminate_on_num_evals:
        if terminate_on_stagnant_best_fitness:
            # Both
            return stagnant_generations >= stagnant_generations_until_termination or num_evals >= evals_per_run
        # Just Evals
        return num_evals >= evals_per_run
    elif terminate_on_stagnant_best_fitness:
        # Just stagnation
        return stagnant_generations >= stagnant_generations_until_termination
    else:
        # Hella problems - idiot user said false for both
        print("Error: Both termination flags set to false!")
        exit(1)

main()
