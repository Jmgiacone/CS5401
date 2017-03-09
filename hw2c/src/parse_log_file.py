from sys import argv


def main():
    if len(argv) < 2:
        print("Error: Too few arguments. Expected 1, found {}".format(len(argv)))
        exit(1)
    elif len(argv) > 2:
        print("Error: Too many arguments. Expected 1, found {}".format(len(argv)))
        exit(1)

    # Exactly one CLI input

    try:
        log_file = open(argv[1], "r")
    except FileNotFoundError:
        print("Error: File \"{}\" not found.".format(argv[1]))
        exit(1)

    run = 0
    index = 0
    x_values = []
    avg_avg_values = []
    avg_best_values = []

    for line in log_file:
        tokens = line.split("\t")

        if tokens[0][:3] == "Run":
            # Next run, reset things
            run += 1
            index = 0
        else:
            try:
                if len(tokens) == 3:
                    if run == 1:
                        x_values.append(int(tokens[0]))
                        avg_avg_values.append(float(tokens[1]))
                        avg_best_values.append(float(tokens[2].strip("\n")))
                    else:
                        avg_avg_values[index] *= (run - 1)
                        avg_avg_values[index] += float(tokens[1])
                        avg_avg_values[index] /= run

                        avg_best_values[index] *= (run - 1)
                        avg_best_values[index] += float(tokens[2].strip("\n"))
                        avg_best_values[index] /= run

                        index += 1

            except ValueError:
                pass

    file_name = argv[1][:argv[1].find(".")] + "_parsed.log"
    output_file = open(file_name, "w")

    for i in range(len(x_values)):
        output_file.write("{}\t{}\t{}\n".format(x_values[i], avg_avg_values[i], avg_best_values[i]))
    print("Successfully wrote to {}".format(file_name))
if __name__ == "__main__":
    main()
