#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <boost/algorithm/string.hpp>
#include <stdlib.h>
#include <cmath>
#include <cfloat>
#include <random>
#include "InitializationChoice.h"
#include "ParentSelection.h"
#include "SurvivalSelection.h"
#include "Genome.h"

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::string;
using std::normal_distribution;

const int NUM_EA_PARAMS = 20;
const float TIMES_TO_EVALUATE_AGAINST_TEST_SET = 10;

float evaluateFitness(const string inputDirectory, const Genome *genome);

Genome *crossover(const Genome *parent1, const Genome *parent2);

void randomGenome(Genome *genome);

void
mutation(Genome *individual, float mutationRate, normal_distribution<double> *rng, std::default_random_engine *gen);

void quickSort(Genome **individuals, const int start, const int end);

int partition(Genome **individuals, int start, int end);

int main(int argc, char *argv[])
{
    //TODO Add in termination parameters
    //General config parameters
    string trainingSet, testSet, solutionFile, trainingLog, testLog;
    int numRuns, evalsPerRun;
    long seed;

    //EA-specific config parameters

    //Chance that mutation will occur [0, 1]
    float mutationRate;

    //Chosen initialization method
    InitializationChoice initChoice;

    //Chosen parent selection method
    ParentSelection parentSelectionChoice;

    //Chosen survival selection method
    SurvivalSelection survivalSelectionChoice;

    int populationSize, offspringPerGeneration, parentSelectionTournamentSize, survivalSelectionTournamentSize,
        numGenerationsUntilStagnation, numEvals = 0;
    bool terminateOnMaxEvals, terminateOnStagnantAverageFitness, terminateOnStagnantBestFitness;

    //Other important things
    float totalFitness = 0, averageTrainingFitness = 0, previousAverageFitness = 0,
          maxFitness = -FLT_MAX, runningProbabilitySum = 0, rValue, bestOverallTestFitness = -FLT_MAX;
    int totalIndividualsProduced = 0, currentGeneration = 0, count, enumTmp, parentsFound = 0,
        tournamentMembersFound = 0, indexOfTournamentWinner, currentMember = 0, iValue = 0, parentChoice, index,
        parentsIndex = 0, individualsSelected, survivalIndex, generationsBetweenTestSetEvals,
        generationsOfNoChangeAverageFitness = 0, generationsOfNoChangeBestFitness = 0;
    bool terminate = false, maxEvalsReached = false, averageFitnessStagnant = false, bestFitnessStagnant = false;
    Genome* bestPopulationMember = nullptr;
    Genome* worstPopulationMember = nullptr;
    Genome* previousBestPopulationMember = nullptr;
    string bestTestSetParams = "";

    normal_distribution<double> *rng = new normal_distribution<double>(0, 1);
    std::default_random_engine *gen = new std::default_random_engine();

    //Usage was not respected (too few or too many arguments)
    if(argc != 2)
    {
        cout << "Usage: " << argv[0] << " <config-file>" << endl;
        return 1;
    }

    ifstream configFileInputStream(argv[1]);

    //Does the file exist?
    if(!configFileInputStream)
    {
        cout << "Error: File \"" << argv[1] << "\" was not found" << endl;
        return 1;
    }

    //File exists - time to parse it
    string line, key, value;
    string eaParams[NUM_EA_PARAMS];

    cout << "=============== Parameters ===============" << endl;
    count = 0;

    //Parse the config file and throw values into a string[]
    while(getline(configFileInputStream, line))
    {
        //Ignore empty lines and lines starting with '#'
        if(line.size() != 0 && line.at(0) != '#')
        {
            std::istringstream lineStringStream(line);

            //Skip the text up until the ':'
            if(getline(lineStringStream, key, ':'))
            {
                if(getline(lineStringStream, value))
                {
                    boost::trim(value);
                    eaParams[count] = value;
                    count++;
                    cout << "#" << count << ") " << key << ": " << value << endl;
                }
            }
        }
    }
    cout << "Read " << count << " values" << endl;

    ofstream solutionFileOut, trainingLogFileOut, testLogFileOut;

    //Set config variables

    //CNF Training input directory
    trainingSet = eaParams[0];

    //CNF Test input directory
    testSet = eaParams[1];

    //Solution File
    solutionFile = eaParams[2];
    solutionFileOut.open(solutionFile);

    //Training Log File
    trainingLog = eaParams[3];
    trainingLogFileOut.open(trainingLog);

    //Test Log File
    testLog = eaParams[4];
    testLogFileOut.open(testLog);

    //Number of runs
    try
    {
        numRuns = std::stoi(eaParams[5]);
    }
    catch(std::invalid_argument e)
    {
        cout << "Error: \"" << eaParams[5] << "\" is not a valid number" << endl;
        return 1;
    }

    //Number of evals per run
    try
    {
        evalsPerRun = std::stoi(eaParams[6]);
    }
    catch(std::invalid_argument e)
    {
        cout << "Error: \"" << eaParams[6] << "\" is not a valid number" << endl;
        return 1;
    }

    //Seed
    if(eaParams[7] == "True" || eaParams[7] == "true")
    {
        //Set seed to current time in micros
        std::chrono::high_resolution_clock::duration duration = std::chrono::high_resolution_clock::now().time_since_epoch();
        seed = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    }
    else
    {
        try
        {
            seed = std::stol(eaParams[7]);
        }
        catch(std::invalid_argument e)
        {
            cout << "Error: \"" << eaParams[7]
                 << "\" is not a valid seed. Please only use seeds from logfiles\n"
                         " or a value of \"true\" if you want a time-generated seed" << endl;
            return 1;
        }
    }
    cout << "Running with seed " << seed << endl;
    std::srand(seed);

    //Population Size
    try
    {
        populationSize = std::stoi(eaParams[8]);
    }
    catch(std::invalid_argument e)
    {
        cout << "Error: \"" << eaParams[8] << "\" is not a valid number" << endl;
        return 1;
    }

    //Offspring per generation
    try
    {
        offspringPerGeneration = std::stoi(eaParams[9]);
    }
    catch(std::invalid_argument e)
    {
        cout << "Error: \"" << eaParams[9] << "\" is not a valid number" << endl;
        return 1;
    }

    //Mutation chance
    try
    {
        mutationRate = std::stof(eaParams[10]);
    }
    catch(std::invalid_argument e)
    {
        cout << "Error: \"" << eaParams[10] << "\" is not a valid number" << endl;
        return 1;
    }

    if(mutationRate < 0 || mutationRate > 1)
    {
        cout << "Error: Mutation rate of " << mutationRate << "is outside acceptable bounds ([0, 1])"
             << endl;
    }

    //Population initialization
    try
    {
        enumTmp = std::stoi(eaParams[11]);
        initChoice = static_cast<InitializationChoice>(enumTmp);
    }
    catch(std::invalid_argument e)
    {
        cout << "Error: \"" << eaParams[11] << "\" is not a valid number" << endl;
        return 1;
    }

    //Parent selection
    try
    {
        enumTmp = std::stoi(eaParams[12]);
        parentSelectionChoice = static_cast<ParentSelection >(enumTmp);
    }
    catch(std::invalid_argument e)
    {
        cout << "Error: \"" << eaParams[12] << "\" is not a valid number" << endl;
        return 1;
    }

    //Parent Selection Tournament size
    try
    {
        parentSelectionTournamentSize = std::stoi(eaParams[13]);
    }
    catch(std::invalid_argument e)
    {
        cout << "Error: \"" << eaParams[13] << "\" is not a valid number" << endl;
        return 1;
    }

    //Survival selection
    try
    {
        enumTmp = std::stoi(eaParams[14]);
        survivalSelectionChoice = static_cast<SurvivalSelection>(enumTmp);
    }
    catch(std::invalid_argument e)
    {
        cout << "Error: \"" << eaParams[14] << "\" is not a valid number" << endl;
        return 1;
    }

    //Survival tournament size
    try
    {
        survivalSelectionTournamentSize = std::stoi(eaParams[15]);
    }
    catch(std::invalid_argument e)
    {
        cout << "Error: \"" << eaParams[15] << "\" is not a valid number" << endl;
        return 1;
    }

    //Max evals termination
    boost::algorithm::to_lower(eaParams[16]);
    terminateOnMaxEvals = (eaParams[16] == "true");

    //Generations until stagnation
    try
    {
        numGenerationsUntilStagnation = std::stoi(eaParams[17]);
    }
    catch(std::invalid_argument e)
    {
        cout << "Error: \"" << eaParams[17] << "\" is not a valid number" << endl;
        return 1;
    }

    //Stagnant average termination
    boost::algorithm::to_lower(eaParams[18]);
    terminateOnStagnantAverageFitness = (eaParams[18] == "true");

    //Stagnant best termination
    boost::algorithm::to_lower(eaParams[19]);
    terminateOnStagnantBestFitness = (eaParams[19] == "true");

    //Start the log file
    time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    trainingLogFileOut << "Training Result Log"
                       << "\n=== Config ==="
                       << "\nStarted at: " << std::ctime(&time)
                       << "CNF Training Set: " << trainingSet
                       << "\nCNF Test Set: " << testSet
                       << "\nSolution File: " << solutionFile
                       << "\nTraining Log File: " << trainingLog
                       << "\nTest Log File: " << testLog
                       << "\nNumber of Runs: " << numRuns
                       << "\nNumber of evals per run: " << evalsPerRun
                       << "\nSeed: " << seed
                       << "\nPopulation Size: " << populationSize
                       << "\nOffspring per generation: " << offspringPerGeneration
                       << "\nMutation Chance: " << mutationRate
                       << "\nInitialization: " << static_cast<int>(initChoice)
                       << "\nParent Selection: " << static_cast<int>(parentSelectionChoice)
                       << "\nParent Selection Tournament Size: " << parentSelectionTournamentSize
                       << "\nSurvival Selection: " << static_cast<int>(survivalSelectionChoice)
                       << "\nSurvival Selection Tournament Size: " << survivalSelectionTournamentSize
                       << "\nTerminate on max evals: " << terminateOnMaxEvals
                       << "\nTerminate on stagnant average fitness: " << terminateOnStagnantAverageFitness
                       << "\nTerminate on stagnant best fitness: " << terminateOnStagnantBestFitness
                       << "\n=== End Config ==="
                       << endl;

    testLogFileOut << "Test Result Log"
                       << "\n=== Config ==="
                       << "\nStarted at: " << std::ctime(&time)
                       << "CNF Training Set: " << trainingSet
                       << "\nCNF Test Set: " << testSet
                       << "\nSolution File: " << solutionFile
                       << "\nTraining Log File: " << trainingLog
                       << "\nTest Log File: " << testLog
                       << "\nNumber of Runs: " << numRuns
                       << "\nNumber of evals per run: " << evalsPerRun
                       << "\nSeed: " << seed
                       << "\nPopulation Size: " << populationSize
                       << "\nOffspring per generation: " << offspringPerGeneration
                       << "\nMutation Chance: " << mutationRate
                       << "\nInitialization: " << static_cast<int>(initChoice)
                       << "\nParent Selection: " << static_cast<int>(parentSelectionChoice)
                       << "\nParent Selection Tournament Size: " << parentSelectionTournamentSize
                       << "\nSurvival Selection: " << static_cast<int>(survivalSelectionChoice)
                       << "\nSurvival Selection Tournament Size: " << survivalSelectionTournamentSize
                       << "\nTerminate on max evals: " << terminateOnMaxEvals
                       << "\nTerminate on stagnant average fitness: " << terminateOnStagnantAverageFitness
                       << "\nTerminate on stagnant best fitness: " << terminateOnStagnantBestFitness
                       << "\n=== End Config ==="
                       << endl;

    //Initialize dynamic memory
    //The population. Stays static at <populationSize> members
    Genome **population = new Genome *[populationSize];

    //The population plus their kids
    Genome **kids = new Genome *[offspringPerGeneration];

    //List to hold parents and children together. Useful for survival selection
    Genome **parentsAndKids = new Genome *[populationSize + offspringPerGeneration];

    //Breeding pool for parents
    Genome **breedingPool = new Genome *[2 * offspringPerGeneration];

    //List to hold parent selection tournaments
    Genome **parentSelectionTournament = new Genome *[parentSelectionTournamentSize];

    //List to hold survival selection tournaments
    Genome **survivalSelectionTournament = new Genome *[survivalSelectionTournamentSize];

    //Array to hold the probability distributions as described in the book
    float *aArray = new float[populationSize];

    //Figure out how many generations to wait before evaluating against the test set
    generationsBetweenTestSetEvals =
            static_cast<int>(std::round(((evalsPerRun - populationSize) / static_cast<float>(offspringPerGeneration))
                             / TIMES_TO_EVALUATE_AGAINST_TEST_SET));

    //Perform <number of runs> runs
    for(int run = 1; run <= numRuns; run++)
    {
        cout << "=============== Run #" << run << " ===============" << endl;
        trainingLogFileOut << "Run " << run << endl;
        testLogFileOut << "Run " << run << endl;

        //Initialize the population
        cout << "=============== Uniform-Random Population initialization ===============" << endl;
        switch(initChoice)
        {
            case InitializationChoice::UNIFORM_RANDOM:
                for(int i = 0; i < populationSize; i++)
                {
                    //Create the memory for the genome
                    population[i] = new Genome();

                    //TODO Reset totalIndividualsProduced every run
                    totalIndividualsProduced++;

                    //Assign it a unique id
                    population[i]->id = totalIndividualsProduced;

                    //Randomize its variables
                    randomGenome(population[i]);

                    //Evaluate its fitness
                    population[i]->trainingFitness = evaluateFitness(trainingSet, population[i]);

                    cout << "ID: " << population[i] -> id << " Fitness: " << population[i]->trainingFitness << endl;

                    //TODO Reset averageTrainingFitness every generation

                    //TODO Reset bestPopulationFitness every run

                    //TODO Reset minFitness every generation

                    //TODO Reset numEvals every run
                    //Keep track of evals
                    numEvals++;
                }
                break;
            default:
                cout << "Invalid initialization method of" << static_cast<int>(initChoice) << endl;
                return 1;
        }

        //Initialize best and worst members
        bestPopulationMember = worstPopulationMember = population[0];

        //Do first set of output
        cout << "=============== Generation #1 ===============" << endl;

        //Output current population
        averageTrainingFitness = 0;
        for(int i = 0; i < populationSize; i++)
        {
            //Sum up training fitness to calculate and average
            averageTrainingFitness += population[i]->trainingFitness;

            cout << "ID: " << population[i]->id << " Fitness: " << population[i]->trainingFitness << endl;

            //Find worst population member for fitness-proportional selection
            if(population[i] -> trainingFitness < worstPopulationMember -> trainingFitness)
            {
                worstPopulationMember = population[i];
            }
                //Find the best population member for test set evaluation as well as training log output
            else if(population[i]->trainingFitness > bestPopulationMember -> trainingFitness)
            {
                bestPopulationMember = population[i];
            }
        }
        averageTrainingFitness /= populationSize;

        previousAverageFitness = averageTrainingFitness;
        previousBestPopulationMember = bestPopulationMember;
        cout << "Best: " << bestPopulationMember -> id << " Worst: " << worstPopulationMember -> id
             << " Average: " << averageTrainingFitness << endl;

        //Write average fitness and best fitness to the log
        trainingLogFileOut << numEvals << "\t" << averageTrainingFitness << "\t" << bestPopulationMember->trainingFitness << endl;

        //Run until termination condition has been met
        //TODO Reset terminate var every run
        while(!terminate)
        {
            //Perform parent selection
            parentsFound = tournamentMembersFound = currentMember = iValue = 0;
            maxFitness = -FLT_MAX;

            switch(parentSelectionChoice)
            {
                case ParentSelection::FITNESS_PROPORTIONAL:
                    cout << "=============== Fitness-Proportional Parent Selection ===============" << endl;

                    //Calculate the sum of the windowed fitnesses
                    for(int i = 0; i < populationSize; i++)
                    {
                        //f'(i) = f(i) - min(f(i))
                        totalFitness += (population[i]->trainingFitness - worstPopulationMember -> trainingFitness);
                    }

                    for(int i = 0; i < populationSize; i++)
                    {
                        //Selection chance = windowed fitness / total
                        population[i]->selectionChance = (population[i]->trainingFitness - worstPopulationMember -> trainingFitness) / totalFitness;

                        //Keep a running total of probability sum
                        runningProbabilitySum += population[i]->selectionChance;

                        //cout << "Running sum of probabilities after " << i+1 << "/" << populationSize << " iterations: " << runningProbabilitySum << endl;
                        //Use this running total to generate the a array
                        aArray[i] = runningProbabilitySum;
                    }

                    //Reset the running sum of probabilities for next generation
                    runningProbabilitySum = 0;

                    //Reset total fitness as well
                    totalFitness = 0;

                    cout << "Choosing " << 2 * offspringPerGeneration << " parents for recombination" << endl;

                    //rand / RAND_MAX -> [0, 1]
                    rValue = (1 / static_cast<float>(2 * offspringPerGeneration)) * static_cast<float>(std::rand()) /
                             RAND_MAX;
                    while(currentMember < 2 * offspringPerGeneration)
                    {
                        while(rValue <= aArray[iValue])
                        {
                            breedingPool[currentMember] = population[iValue];

                            cout << "ID: " << breedingPool[currentMember]->id << " Fitness: "
                                 << breedingPool[currentMember]->trainingFitness << endl;
                            rValue += (1 / static_cast<float>(2 * offspringPerGeneration));

                            currentMember++;
                        }
                        iValue++;
                    }

                    //Reset important values
                    iValue = 0;
                    rValue = 0;
                    currentMember = 0;
                    break;
                case ParentSelection::TOURNAMENT_SELECTION:
                    cout << "=============== " << parentSelectionTournamentSize
                         << "-Tournament Selection w/ replacement ===============" << endl;

                    //Find 2*lambda parents and throw them into the breeding pool
                    while(parentsFound != 2 * offspringPerGeneration)
                    {
                        //Reset number of members found
                        tournamentMembersFound = 0;

                        cout << "=== Finding tournament attendees ===" << endl;
                        //Randomly pull <tournamentSize> members from the population and put them in a tournament
                        while(tournamentMembersFound != parentSelectionTournamentSize)
                        {
                            //Choose a random member of the population
                            parentChoice = std::rand() % populationSize;

                            //If they're not already in the tournament, throw them in
                            if(population[parentChoice] != nullptr && !population[parentChoice]->enteredInTournament)
                            {
                                parentSelectionTournament[tournamentMembersFound] = population[parentChoice];
                                tournamentMembersFound++;
                                cout << "ID: " << population[parentChoice]->id << " Fitness: "
                                     << population[parentChoice]->trainingFitness << endl;
                                population[parentChoice]->enteredInTournament = true;
                            }
                        }


                        //Reset pertinent vars
                        indexOfTournamentWinner = 0;
                        maxFitness = -FLT_MAX;
                        tournamentMembersFound = 0;

                        //Find the winner of this tournament
                        for(int i = 0; i < parentSelectionTournamentSize; i++)
                        {
                            parentSelectionTournament[i]->enteredInTournament = false;
                            if(parentSelectionTournament[i]->trainingFitness > maxFitness)
                            {
                                maxFitness = parentSelectionTournament[i]->trainingFitness;
                                indexOfTournamentWinner = i;
                            }
                        }

                        cout << "Winner: " << parentSelectionTournament[indexOfTournamentWinner]->id << " Fitness: "
                             << parentSelectionTournament[indexOfTournamentWinner]->trainingFitness << endl;

                        //Place the winner in the breeding pool
                        breedingPool[parentsFound] = parentSelectionTournament[indexOfTournamentWinner];
                        parentsFound++;
                    }
                    parentsFound = 0;
                    break;
                default:
                    cout << "Invalid parent selection method of " << static_cast<int>(parentSelectionChoice) << endl;
                    return 1;
            }

            cout << "=============== Recombination ===============" << endl;
            for(int i = 0; i < 2 * offspringPerGeneration; i++)
            {
                cout << "ID: " << breedingPool[i]->id << " Fitness: " << breedingPool[i]->trainingFitness << endl;
            }
            //Take the parents, pair them randomly and perform crossover
            Genome *parents[2];

            parents[0] = parents[1] = nullptr;
            parentsIndex = 0;
            for(int i = 0; i < offspringPerGeneration; i++)
            {
                //Choose 2 parents
                while(!(parents[0] != nullptr && parents[1] != nullptr))
                {
                    index = std::rand() % (2 * offspringPerGeneration);

                    //Parent hasn't been chosen yet
                    if(breedingPool[index] != nullptr)
                    {
                        parents[parentsIndex] = breedingPool[index];
                        parents[parentsIndex]->enteredInTournament = false;
                        breedingPool[index] = nullptr;
                        parentsIndex++;
                    }
                }
                parentsIndex = 0;

                //Crossover
                kids[i] = crossover(parents[0], parents[1]);
                totalIndividualsProduced++;
                kids[i]->id = totalIndividualsProduced;

                //Mutation
                mutation(kids[i], mutationRate, rng, gen);

                //Evaluate fitness
                kids[i]->trainingFitness = evaluateFitness(trainingSet, kids[i]);
                cout << "Individual " << kids[i]->id << " (child of " << parents[0]->id << " and " << parents[1]->id
                     << ") Fitness: " << kids[i]->trainingFitness << endl;

                numEvals++;
                parents[0] = parents[1] = nullptr;
            }

            //Throw everybody into one big pool and do survival selection
            for(int i = 0; i < populationSize; i++)
            {
                parentsAndKids[i] = population[i];
                population[i] = nullptr;
            }

            for(int i = 0; i < offspringPerGeneration; i++)
            {
                parentsAndKids[populationSize + i] = kids[i];
                kids[i] = nullptr;
            }

            switch(survivalSelectionChoice)
            {
                case SurvivalSelection::TOURNAMENT_SELECTION:
                    cout << "=============== " << survivalSelectionTournamentSize
                         << "-Tournament Survival Selection w/o replacement ===============" << endl;
                    individualsSelected = survivalIndex = 0;

                    while(individualsSelected != populationSize)
                    {
                        //Select <survival tournament size> individuals and hold a tournament
                        tournamentMembersFound = 0;

                        cout << "=== Finding tournament attendees ===" << endl;
                        while(tournamentMembersFound != survivalSelectionTournamentSize)
                        {
                            survivalIndex = std::rand() % (populationSize + offspringPerGeneration);

                            //If the individual isn't entered in a tournament already
                            if(parentsAndKids[survivalIndex] != nullptr &&
                               !parentsAndKids[survivalIndex]->enteredInTournament)
                            {
                                survivalSelectionTournament[tournamentMembersFound] = parentsAndKids[survivalIndex];
                                cout << "ID: " << survivalSelectionTournament[tournamentMembersFound]->id
                                     << " Fitness: "
                                     << survivalSelectionTournament[tournamentMembersFound]->trainingFitness << endl;
                                survivalSelectionTournament[tournamentMembersFound]->enteredInTournament = true;
                                tournamentMembersFound++;
                            }
                        }

                        //Determine the winner
                        maxFitness = -FLT_MAX;
                        for(int i = 0; i < survivalSelectionTournamentSize; i++)
                        {
                            survivalSelectionTournament[i]->enteredInTournament = false;
                            if(survivalSelectionTournament[i]->trainingFitness > maxFitness)
                            {
                                maxFitness = survivalSelectionTournament[i]->trainingFitness;
                                indexOfTournamentWinner = i;
                            }
                        }

                        individualsSelected++;
                        cout << "Winner: " << individualsSelected << " ID: "
                             << survivalSelectionTournament[indexOfTournamentWinner]->id << endl;
                        //Take the winner and secure its spot in the next generation
                        for(int i = 0; i < populationSize + offspringPerGeneration; i++)
                        {
                            if(parentsAndKids[i] == survivalSelectionTournament[indexOfTournamentWinner])
                            {
                                //Move the winner to the general population
                                population[individualsSelected - 1] =
                                        survivalSelectionTournament[indexOfTournamentWinner];
                                parentsAndKids[i] = nullptr;
                            }
                        }
                    }

                    break;
                case SurvivalSelection::TRUNCATION:
                    cout << "=============== Truncation Survival Selection ===============" << endl;

                    //Sort whole population in descending fitness and take the top <population size>
                    quickSort(parentsAndKids, 0, populationSize + offspringPerGeneration - 1);

                    for(int i = 0; i < populationSize; i++)
                    {
                        cout << "Moving " << parentsAndKids[i]->id << " to the next generation" << endl;
                        population[i] = parentsAndKids[i];
                        parentsAndKids[i] = nullptr;
                    }
                    break;
                default:
                    cout << "Invalid survival selection method of " << static_cast<int>(survivalSelectionChoice)
                         << endl;
                    return 1;
            }

            //Kill the individuals that weren't selected
            for(int i = 0; i < populationSize + offspringPerGeneration; i++)
            {
                if(parentsAndKids[i] != nullptr)
                {
                    cout << "Killing individual " << parentsAndKids[i]->id << endl;
                    delete parentsAndKids[i];
                    parentsAndKids[i] = nullptr;
                }
            }

            //Find best member and calculate average fitness
            averageTrainingFitness = 0;
            for(int i = 0; i < populationSize; i++)
            {
                //Sum up training fitness to calculate and average
                averageTrainingFitness += population[i]->trainingFitness;

                cout << "ID: " << population[i]->id << " Fitness: " << population[i]->trainingFitness << endl;

                //Find worst population member for fitness-proportional selection
                if(population[i] -> trainingFitness < worstPopulationMember -> trainingFitness)
                {
                    worstPopulationMember = population[i];
                }
                    //Find the best population member for test set evaluation as well as training log output
                else if(population[i]->trainingFitness > bestPopulationMember -> trainingFitness)
                {
                    bestPopulationMember = population[i];
                }
            }
            averageTrainingFitness /= populationSize;

            cout << "Best: " << bestPopulationMember -> id << " Worst: " << worstPopulationMember -> id
                 << " Average: " << averageTrainingFitness << endl;

            //Write average fitness and best fitness to the log
            trainingLogFileOut << numEvals << "\t" << averageTrainingFitness << "\t" << bestPopulationMember->trainingFitness << endl;

            cout << "Evals: " << numEvals << " Previous Best: " << previousBestPopulationMember->trainingFitness
                 << " Current Best: " << bestPopulationMember -> trainingFitness << " Previous Average: "
                 << previousAverageFitness << " Current Average: " << averageTrainingFitness << endl;

            //Test to see if we should evaluate the best member against the test set
            if(currentGeneration % generationsBetweenTestSetEvals == 0)
            {
                cout << "Evaluating best member against test set" << endl;

                bestPopulationMember -> testFitness = evaluateFitness(testSet, bestPopulationMember);

                if(bestPopulationMember -> testFitness > bestOverallTestFitness)
                {
                    bestOverallTestFitness = bestPopulationMember -> testFitness;
                    bestTestSetParams = bestPopulationMember -> getParamString();
                }

                testLogFileOut << numEvals << "\t" << bestPopulationMember -> testFitness << endl;
                cout << "ID: " << bestPopulationMember -> id << " Test Fitness: "
                     << bestPopulationMember -> testFitness << endl;
            }

            //Determine if conditions are true
            maxEvalsReached = (numEvals >= evalsPerRun);

            //Is average fitness stagnant?
            if(previousAverageFitness == averageTrainingFitness)
            {
                generationsOfNoChangeAverageFitness++;
                cout << "Average has been stagnant for " << generationsOfNoChangeAverageFitness << " generations"
                     << endl;
            }
            else
            {
                generationsOfNoChangeAverageFitness = 0;
            }
            previousAverageFitness = averageTrainingFitness;
            averageFitnessStagnant = (generationsOfNoChangeAverageFitness >= numGenerationsUntilStagnation);

            //Is best fitness stagnant?
            if(previousBestPopulationMember == bestPopulationMember)
            {
                generationsOfNoChangeBestFitness++;
                cout << "Best has been stagnant for " << generationsOfNoChangeBestFitness << " generations" << endl;
            }
            else
            {
                generationsOfNoChangeBestFitness = 0;
            }
            previousBestPopulationMember = bestPopulationMember;
            bestFitnessStagnant = (generationsOfNoChangeBestFitness >= numGenerationsUntilStagnation);

            //Test termination conditions
            if(terminateOnMaxEvals)
            {
                if(terminateOnStagnantAverageFitness)
                {
                    if(terminateOnStagnantBestFitness)
                    {
                        //All 3 selected
                        terminate = maxEvalsReached || averageFitnessStagnant || bestFitnessStagnant;
                    }
                    else
                    {
                        //Terminate on max evals and stagnant average, but not stagnant best
                        terminate = maxEvalsReached || averageFitnessStagnant;
                    }
                }
                else
                {
                    if(terminateOnStagnantBestFitness)
                    {
                       //Terminate on max evals and stagnant best
                       terminate = maxEvalsReached || bestFitnessStagnant;
                    }
                    else
                    {
                        //Terminate on max evals only
                        terminate = maxEvalsReached;
                    }
                }
            }
            else if(terminateOnStagnantAverageFitness)
            {
                if(terminateOnStagnantBestFitness)
                {
                    //Terminate on stagnant average and stagnant best
                    terminate = averageFitnessStagnant || bestFitnessStagnant;
                }
                else
                {
                    //Only terminate on stagnant average
                    terminate = averageFitnessStagnant;
                }
            }
            else if(terminateOnStagnantBestFitness)
            {
                //Terminate on stagnant best only
                terminate = bestFitnessStagnant;
            }
            else
            {
                //Never stop
                terminate = false;
            }

            //Reset average fitness to prepare it for later
            totalFitness = 0;
            currentGeneration++;

            if(!terminate)
            {
                cout << "============ Generation #" << currentGeneration+1 << " ===============" << endl;
            }
        }

        averageTrainingFitness = 0;
        numEvals = 0;
        currentGeneration = 0;
        terminate = false;
        totalIndividualsProduced = 0;

        bestPopulationMember = previousBestPopulationMember = worstPopulationMember = nullptr;
        //Delete the whole population
        for(int i = 0; i < populationSize; i++)
        {
            if(population[i] != nullptr)
            {
                delete population[i];
                population[i] = nullptr;
            }
        }

        //Null out the kids list because everyone was already taken care of in the population deletion
        for(int i = 0; i < offspringPerGeneration; i++)
        {
            kids[i] = nullptr;
        }

        trainingLogFileOut << endl;
        testLogFileOut << endl;
    }

    //Clean up dynamic memory
    delete[] kids;
    delete[] population;
    delete[] breedingPool;
    delete[] parentSelectionTournament;
    delete[] survivalSelectionTournament;
    delete[] parentsAndKids;
    delete[] aArray;

    delete rng;
    delete gen;

    //Close file streams
    solutionFileOut << "./solvers/minisat/minisat " << bestTestSetParams << " " << testSet << endl;

    time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    trainingLogFileOut << "Ended at " << ctime(&time);
    testLogFileOut << "Ended at " << ctime(&time);
    trainingLogFileOut.close();
    testLogFileOut.close();
    solutionFileOut.close();
    configFileInputStream.close();
    return 0;
}

float evaluateFitness(const string inputDirectory, const Genome *genome)
{
    int exit;
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<float> duration;

    string command = "./solvers/minisat/minisat " + genome->getParamString() + " " + inputDirectory + " >> /dev/null";

    start = std::chrono::high_resolution_clock::now();
    exit = system(command.c_str());
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::duration<float>>(end - start);

    if(exit != 0)
    {
        cout << "Non-zero exit code!";
        return -FLT_MAX;
    }

    return 100 - duration.count();






    /*float elapsedTime = -FLT_MAX;
    int exitCode;
    string s;
    string command1 = "./solvers/minisat/minisat " + genome->getParamString() + " " + inputDirectory + " > output.txt";
    string command2 = "cat output.txt | tail -3 | head -1 | cut -f17 -d ' ' > file.txt";
    exitCode = system(command1.c_str());
    system(command2.c_str());

    if(exitCode != 0)
    {
        cout << "Error: Nonzero exit code!";
        system("echo $'=====' >> error.log");
        system("cat output.txt >> error.log");
        return -FLT_MAX;
    }
    std::ifstream inputFile;
    inputFile.open("file.txt");

    if(inputFile)
    {
        //Check to see if the file is empty
        if(inputFile.peek() != std::ifstream::traits_type::eof())
        {
            inputFile >> s;

            try
            {
                elapsedTime = stof(s);
            }
            catch(std::invalid_argument e)
            {
                cout << "Oops... " << s << " is not what we were looking for" << endl;
                system("echo $'=====' >> error.log");
                system("cat output.txt >> error.log");
                return -FLT_MAX;
            }
        }
        else
        {
            cout << "Empty file!";
            return -FLT_MAX;
        }

        do
        {
            inputFile.close();
        }
        while(inputFile.is_open());
    }

    return 100 - elapsedTime;*/
}

void randomGenome(Genome *genome)
{
    //rand() / RAND_MAX -> gives a double in the range [0, 1]

    //luby or no-luby (in set {0, 1})
    genome->luby = std::rand() % 2 == 1;

    //rnd-freq -> [0, 1]
    genome->rnd_freq = static_cast<double>(std::rand()) / RAND_MAX;

    //var-decay -> (0, 1)
    genome->var_decay = static_cast<double>(std::rand()) / RAND_MAX;

    //Make the range exclusive
    if(genome->var_decay == 0)
    {
        genome->var_decay = .0001;
    }
    else if(genome->var_decay == 1)
    {
        genome->var_decay = .9999;
    }

    //cla-decay -> (0, 1)
    genome->cla_decay = static_cast<double>(std::rand()) / RAND_MAX;

    //Make the ranges exclusive
    if(genome->cla_decay == 0)
    {
        genome->cla_decay = .0001;
    }
    else if(genome->cla_decay == 1)
    {
        genome->cla_decay = .9999;
    }

    //rinc -> (1, inf)
    genome->rinc = (DBL_MAX * (static_cast<double>(std::rand()) / RAND_MAX)) + 1.0001;

    //gc-frac -> (0, inf)
    genome->gc_frac = (DBL_MAX - .0001) * (static_cast<double>(std::rand()) / RAND_MAX) + .0001;

    //rfirst -> [1, INT_MAX]
    genome->rfirst = (INT_MAX - 1) * (static_cast<double>(std::rand()) / RAND_MAX) + 1;

    //ccmin-mode -> {0, 1, 2}
    genome->ccmin_mode = std::rand() % 3;

    //phase-saving -> {0, 1, 2}
    genome->phase_saving = std::rand() % 3;
}

Genome *crossover(const Genome *parent1, const Genome *parent2)
{
    //Standard coin flip crossover
    int coinFlips[9];
    const Genome *parents[2];
    parents[0] = parent1;
    parents[1] = parent2;

    for(int i = 0; i < 9; i++)
    {
        coinFlips[i] = std::rand() % 2;
    }

    Genome *child = new Genome();

    child->luby = parents[coinFlips[0]]->luby;
    child->rnd_freq = parents[coinFlips[1]]->rnd_freq;
    child->var_decay = parents[coinFlips[2]]->var_decay;
    child->cla_decay = parents[coinFlips[3]]->cla_decay;
    child->rinc = parents[coinFlips[4]]->rinc;
    child->gc_frac = parents[coinFlips[5]]->gc_frac;
    child->rfirst = parents[coinFlips[6]]->rfirst;
    child->ccmin_mode = parents[coinFlips[7]]->ccmin_mode;
    child->phase_saving = parents[coinFlips[8]]->phase_saving;

    return child;
}

void mutation(Genome *individual, float mutationRate, normal_distribution<double> *rng, std::default_random_engine *gen)
{
    //Standard coin flip crossover
    float randomNumbers[9];
    double x;
    int y;

    for(int i = 0; i < 9; i++)
    {
        randomNumbers[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    //If mutation occurs, randomize their luby value
    individual->luby = randomNumbers[0] <= mutationRate ? std::rand() % 2 == 1 : individual->luby;

    //If mutation occurs, add a normally distributed value to rnd_freq
    if(randomNumbers[1] <= mutationRate)
    {
        individual->rnd_freq += rng->operator()(*gen);

        individual->rnd_freq = individual->rnd_freq < 0 ? 0 : individual->rnd_freq;
        individual->rnd_freq = individual->rnd_freq > 1 ? 1 : individual->rnd_freq;
    }

    //If mutation occurs, add a normally distributed value to var_decay
    if(randomNumbers[2] <= mutationRate)
    {
        individual->var_decay += rng->operator()(*gen);

        individual->var_decay = individual->var_decay <= 0 ? .0001 : individual->var_decay;
        individual->var_decay = individual->var_decay >= 1 ? .9999 : individual->var_decay;
    }

    //If mutation occurs, add a normally distributed value to cla_decay
    if(randomNumbers[3] <= mutationRate)
    {
        individual->cla_decay += rng->operator()(*gen);

        individual->cla_decay = individual->cla_decay <= 0 ? .0001 : individual->cla_decay;
        individual->cla_decay = individual->cla_decay >= 1 ? .9999 : individual->cla_decay;
    }

    //If mutation occurs, add a normally distributed value to rinc
    if(randomNumbers[4] <= mutationRate)
    {
        x = DBL_MAX * rng->operator()(*gen);

        //If adding x will overflow a double
        if(individual->rinc + x > DBL_MAX)
        {
            individual->rinc = DBL_MAX;
        }
        else if(individual->rinc + x < -DBL_MAX)
        {
            individual->rinc = 1.0001;
        }
        else
        {
            individual->rinc += x;
            individual->rinc = individual->rinc + x <= 1 ? 1.0001 : individual->rinc;
        }
    }

    //If mutation occurs, add a normally distributed value to gc_frac
    if(randomNumbers[5] <= mutationRate)
    {
        x = DBL_MAX * rng->operator()(*gen);

        //If adding x will overflow a double
        if(individual->gc_frac + x > DBL_MAX)
        {
            individual->gc_frac = DBL_MAX;
        }
        else if(individual->gc_frac + x < -DBL_MAX)
        {
            individual->gc_frac = .0001;
        }
        else
        {
            individual->gc_frac += x;
            individual->gc_frac = individual->gc_frac + x <= 0 ? .0001 : individual->gc_frac;
        }
    }

    //If mutation occurs, add a normally distributed value to rfirst
    if(randomNumbers[6] <= mutationRate)
    {
        y = static_cast<int>(INT_MAX * rng->operator()(*gen));

        //If adding y overflows an int
        if(individual->rfirst + y > INT_MAX)
        {
            individual->rfirst = INT_MAX;
        }
        else if(individual->rfirst + y < -INT_MAX)
        {
            individual->rfirst = 1;
        }
        else
        {
            individual->rfirst += y;
            individual->rfirst = individual->rfirst < 1 ? 1 : individual->rfirst;
        }
    }

    //If mutation occurs, randomize their ccmin_mode value
    individual->ccmin_mode = randomNumbers[7] <= mutationRate ? std::rand() % 3 : individual->ccmin_mode;

    //If mutation occurs, randomize their phase_saving value
    individual->phase_saving = randomNumbers[8] <= mutationRate ? std::rand() % 3 : individual->phase_saving;
}

int partition(Genome **individuals, int start, int end)
{
    int pivot, i = start - 1;
    Genome *pivotGenome = individuals[end];
    Genome *temp;

    for(int j = start; j < end; j++)
    {
        if(individuals[j]->trainingFitness >= pivotGenome->trainingFitness)
        {
            i++;
            temp = individuals[j];

            individuals[j] = individuals[i];

            individuals[i] = temp;
        }
    }
    pivot = i + 1;

    temp = individuals[end];

    individuals[end] = individuals[pivot];

    individuals[pivot] = temp;

    return pivot;
}

void quickSort(Genome **individuals, const int start, const int end)
{
    int p;
    if(start < end)
    {
        p = partition(individuals, start, end);
        quickSort(individuals, start, p - 1);
        quickSort(individuals, p + 1, end);
    }
}