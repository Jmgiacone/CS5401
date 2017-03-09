import sys
import configparser
import copy
from random import randrange
from random import randint
from random import seed
from os.path import exists
from os import makedirs


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
        random_cell = find_empty_space(board, "#")

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


def build_initial_world_file(board, file, initial_time):
    file.write(str(len(board[0])) + "\n")
    file.write(str(len(board)) + "\n")
    for row in range(len(board)):
        for column in range(len(board[row])):
            if board[row][column] == "#":
                file.write("w {} {}\n".format(column, row))
            elif board[row][column] == "M":
                file.write("m {} {}\n".format(column, row))
            elif board[row][column] == "G":
                file.write("1 {} {}\n".format(column, row))
                file.write("2 {} {}\n".format(column, row))
                file.write("3 {} {}\n".format(column, row))
            elif board[row][column] == "*":
                file.write("p {} {}\n".format(column, row))

    file.write("t {} 0\n".format(initial_time))


def print_board(board):
    for row in range(len(board)):
        for column in range(len(board[row])):
            print(board[row][column], end=" ")
        print()


def play_game(rows, columns, wall_density, pill_density, fruit_chance, fruit_score, time_multiplier, world_file):
    # Initialize pertinent book-keeping info
    pac_man_position = None
    pac_man_previous_position = ()
    ghost_positions = []
    ghost_previous_positions = []
    fruit_position = None
    game_over = False
    pills_collected = 0
    fruit_collected = 0
    score = 0
    time_remaining = rows * columns * time_multiplier
    pac_man_dead = False
    fruit_written = False  # Flag for whether or not current fruit has been written to the file
    num_pills = 0  # Number of pills on the board

    board = ["#"] * rows
    for b in range(rows):
        board[b] = ["#"] * columns

    walls = prims_maze_generator(board, wall_density)

    # Assign Pac-Man a start spot
    while pac_man_position is None:
        pac_man_position = find_empty_space(board)

    # Place an 'M' in pac-man's spot
    board[pac_man_position[0]][pac_man_position[1]] = "M"

    # Assign the ghosts a start spot (all 3 start in the same spot)
    ghosts_start = None
    while ghosts_start is None:
        ghosts_start = find_empty_space(board)

    # Place a 'G' for ghost start spot
    board[ghosts_start[0]][ghosts_start[1]] = "G"
    ghost_positions = [ghosts_start] * 3

    expected_pills = max(1, int(pill_density * (rows * columns - 2 - walls)))

    # While there are open spaces, place down pills
    while num_pills < expected_pills:
        random_cell = find_empty_space(board)

        if random_cell is None:
            continue

        board[random_cell[0]][random_cell[1]] = "*"
        num_pills += 1

    build_initial_world_file(board, world_file, time_remaining)

    # Play the game
    while not game_over:
        # Attempt to spawn fruit
        if fruit_position is None and randrange(100) + 1 < fruit_chance:
            # Spawn a fruit in the maze
            fruit_position = find_empty_space(board, ".")

            if fruit_position is not None:
                fruit_written = False
                board[fruit_position[0]][fruit_position[1]] = "f"

        # Keep track of pac-man's previous position
        pac_man_previous_position = pac_man_position

        # Move pac-man
        pac_man_position = move_pac_man(board, pac_man_position)

        # Move ghosts
        ghost_previous_positions = copy.deepcopy(ghost_positions)  # Deep copy is necessary so that the previous
        #                                                            positions don't change when the current ones do

        # Move each of the 3 ghosts
        for i in range(3):
            ghost_positions[i] = move_ghost(board, ghost_positions[i])

        # Check to see if pac-man was killed by a ghost
        if pac_man_position in ghost_positions:
            # Pac-man shares a spot with a ghost, game over
            pac_man_dead = True
            game_over = True
        elif pac_man_position in ghost_previous_positions and pac_man_previous_position in ghost_positions:
            # Pac-man swapped spots with a ghost, game over
            pac_man_dead = True
            game_over = True

        # If pac-man survived the movement phase
        if not pac_man_dead:
            # Check pills
            if board[pac_man_position[0]][pac_man_position[1]] == "*":
                pills_collected += 1
                score = pills_collected * 100 // num_pills + fruit_collected * fruit_score

            # Check for fruit
            if pac_man_position == fruit_position:
                fruit_collected += 1
                score += fruit_score
                fruit_position = None

            # Update the board to reflect movement
            board[pac_man_previous_position[0]][pac_man_previous_position[1]] = "."
            board[pac_man_position[0]][pac_man_position[1]] = "M"

            # Decrement remaining time
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

        world_file.write("m {} {}\n".format(pac_man_position[1], pac_man_position[0]))

        if fruit_position is not None and not fruit_written:
            world_file.write("f {} {}\n".format(fruit_position[1], fruit_position[0]))
            fruit_written = True

        for i in range(len(ghost_positions)):
            world_file.write("{} {} {}\n".format(i + 1, ghost_positions[i][1], ghost_positions[i][0]))

        world_file.write("t {} {}\n".format(time_remaining, score))

    return score


def move_pac_man(board, pac_man_position):
    """
    :rtype: tuple(int, int)
    """
    possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]

    while True:
        random_move = possible_moves[randrange(5)]
        r = random_move[0] + pac_man_position[0]
        c = random_move[1] + pac_man_position[1]

        try:
            if r >= 0 and c >= 0 and board[r][c] != "#":
                return r, c
        except IndexError:
            pass


def move_ghost(board, ghost_position):
    """
    :rtype: tuple(int, int)
    """
    possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while True:
        random_move = possible_moves[randrange(4)]
        r = random_move[0] + ghost_position[0]
        c = random_move[1] + ghost_position[1]

        try:
            if r >= 0 and c >= 0 and board[r][c] != "#":
                return r, c
        except IndexError:
            pass


def find_empty_space(board, empty_space_char="."):
    """
    :rtype: tuple(int, int)
    """
    rows, columns = len(board), len(board[0])

    space = randrange(rows), randrange(columns)

    if board[space[0]][space[1]] != empty_space_char:
        return None

    return space


def main():
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
    num_evals = int(config["General Config"]["Fitness Evals Per Run"])
    rows, columns = int(config["World Config"]["World Height"]), int(config["World Config"]["World Width"])
    wall_density = float(config["World Config"]["Wall Density"]) / 100
    pill_density = float(config["World Config"]["Pill Density"]) / 100
    fruit_chance = 100 * float(config["World Config"]["Fruit Spawning Probability"])
    fruit_score = int(config["World Config"]["Fruit Score"])
    time_multiplier = int(config["World Config"]["Time Multiplier"])
    game_seed = config["General Config"]["Seed"]

    # Test the seed variable. If it's an int, then seed with it, otherwise seed with a random seed
    try:
        game_seed = int(game_seed)
        seed(game_seed)
    except ValueError:
        game_seed = randint(0, sys.maxsize)
        seed(game_seed)

    # Open log file
    log_file = open(str(config["Logging Config"]["Log File"]), "w+")
    log_file.write("Result Log\n")

    config_string = ""
    # Aggregate config to a string
    for section in config.sections():
        for (key, value) in config.items(section):
            if key != "seed":
                config_string += "{} = {}\n".format(key, value)
            else:
                config_string += "seed = {}\n".format(game_seed)

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

    print("=== Config ===")
    print(config_string, end="")
    print("=== End Config ===")
    for run in range(num_runs):
        max_fitness = -1

        # Write run number to log file
        log_file.write("Run {}\n".format(run + 1))

        print("Run {}".format(run + 1))

        if not exists(world_file_path + "/" + "run" + str(run + 1)):
            makedirs(world_file_path + "/" + "run" + str(run + 1))

        for game in range(num_evals):
            # Open the world file
            world_file = open(world_file_path + "/" + "run" + str(run + 1) + "/" + str(game + 1) + ".world", "w+")

            # Play the game
            fitness = play_game(rows, columns, wall_density, pill_density, fruit_chance, fruit_score, time_multiplier,
                                world_file)

            # Close the world file
            world_file.close()

            # Check if this fitness is the best overall
            if fitness > max_fitness:
                log_file.write("{}\t{}\n".format(game + 1, fitness))
                print("Game {}: {}".format(game + 1, fitness))
                max_fitness = fitness
            else:
                print("(Game {}: {})".format(game + 1, fitness))

        log_file.write("\n")
        print()

    # Close the log file
    log_file.close()


if __name__ == "__main__":
    main()
