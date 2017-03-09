from random import randrange


def draw_wall_in_direction(board, start_point, direction, length):
    row = start_point[0]
    column = start_point[1]

    for i in range(length):
        try:
            if not (row <= 0 or column <= 0) and board[row][column] == "#":
                return
            board[row][column] = "#"
        except IndexError:
            return

        row += direction[0]
        column += direction[1]


def select_wall_start(rows, columns, granularity):
    return granularity * randrange(int(rows / granularity)), granularity * randrange(int(columns / granularity))


def attempt_random_wall(board, min_length, max_length, granularity):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    chosen_spot = select_wall_start(len(board), len(board[0]), granularity)

    if board[chosen_spot[0]][chosen_spot[1]] == "#":
        return

    direction = directions[randrange(4)]
    length = randrange(min_length, max_length + 1, 1)

    length = length * granularity + 1

    draw_wall_in_direction(board, chosen_spot, direction, length)


def generate_maze_game_dev(board, min_length, max_length, granularity, wall_density):
    """http://members.gamedev.net/vertexnormal/tutorial_randlev1.html"""
    rows, columns = len(board), len(board[0])
    expected_walls = int(wall_density * rows * columns)

    for i in range(expected_walls):
        attempt_random_wall(board, min_length, max_length, granularity)


