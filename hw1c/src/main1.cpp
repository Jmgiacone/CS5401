#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <assert.h>
#include <chrono>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <random>

#include "Genome.h"
#include "InitializationMethod.h"
#include "ParentSelectionMethod.h"
#include "SurvivalSelectionMethod.h"
#include "SurvivalStrategy.h"

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::ifstream;
using std::ofstream;

void parseConfig(ifstream& configFileIn, boost::program_options::variables_map& configVm);
void initializePopulation(Genome** population, int populationSize, InitializationMethod initializationMethod[9],
                          string trainingSet);
double evaluateFitness(string inputDirectory, Genome* individual);
void parentSelection(Genome** population, const int populationSize, Genome** breedingPool,
                     const int parentSelectionPoolSize, const int parentSelectionTournamentSize,
                     ParentSelectionMethod parentSelectionMethod);
void recombination(Genome** breedingPool, const int breedingPoolSize, Genome** kids, const int numKids,
                   string trainingSet);
Genome* crossover(const Genome* parent1, const Genome* parent2);
void mutateIndividual(Genome* individual, double mutationRate, std::normal_distribution<double> *rng,
                      std::default_random_engine *gen);
void survivalSelection(Genome** entirePopulation, const int populationSize, const int numKids,
                       const int survivalSelectionTournamentSize, SurvivalSelectionMethod survivalSelectionMethod);
bool terminationConditionMet();
bool populationHasConverged();
void randomGenome(Genome* individual);
void checkParamRanges(const Genome* genome);
void tournamentSelection(Genome** population, const int populationSize, Genome** tournamentWinners,
                         const int numWinnersToChoose, int tournamentSize, bool withReplacement);
void fitnessProportionalSelection(Genome** population, const int populationSize, Genome** selectedIndividuals,
                                  const int numIndividualsToChoose);
int partition(Genome** population, int startIndex, int endIndex);
void quicksort(Genome** population, int startIndex, int endIndex);
Genome* findExtremeIndividual(Genome** population, const int populationSize, const bool best);
double findAverageFitness(Genome** population, const int populationSize);
void printLineInLog(ofstream& logFileOut, Genome** population, const int populationSize);
void uniformRandomInitialization(Genome* individual, int geneIndex);
void biasedInitialization(Genome* individual, int geneIndex, InitializationMethod initBias);
Genome* readInIndividual(ifstream& fileIn);

//Book-keeping parameters
int evalsPerRun, totalIndividualsProduced = 0, numEvals = 0, numGenerationsUntilStagnation,
    generationsAverageStagnant = 0, generationsBestStagnant = 0;
double mutationRate, previousAverageFitness, previousBestFitness, currentAverageFitness, bestOverallTestFitness = 0;
string absoluteBestParamString;
Genome* bestPopulationMember;
Genome* worstPopulationMember;
bool terminateOnMaxEvals, terminateOnStagnantAverageFitness, terminateOnStagnantBestFitness, justRestarted = false,
     selfAdaptiveMutation, readInBestIndividuals = false;
int populationSize, numIndividualsToReadIn;
ifstream bestIndividualsFileIn;
std::normal_distribution<double> *rng = new std::normal_distribution<double>(0, .25);
std::default_random_engine *gen = new std::default_random_engine();

const int TIMES_TO_EVALUATE_AGAINST_TEST_SET = 10;
int main(int argc, const char* argv[])
{
  boost::program_options::variables_map configVm;
  ofstream testLogFileOut, trainingLogFileOut, solutionFileOut, lastPopulationFileOut;
  string configString;
  int generation = 1;

  //Important EA params
  ParentSelectionMethod parentSelectionMethod;
  SurvivalStrategy survivalStrategy;
  SurvivalSelectionMethod survivalSelectionMethod;
  string trainingSet, testSet, solutionFile, trainingLog, testLog;
  int numRuns, offspringPerGeneration, parentSelectionTournamentSize,
      survivalSelectionTournamentSize;
  long seed;
  bool restartsEnabled;
  int rRestartValue;
  int generationsBetweenTestSetEvals;
  InitializationMethod genesInitMethods[9];
  string bestIndividualsFileName;

  //Too many/too few parameters given
  if(argc != 2)
  {
    cerr << "Usage: " << argv[0] << "<config file>" << endl;
    return 1;
  }

  //Correct number of params given
  ifstream configFileIn(argv[1]);

  //Is the parameter a valid file?
  if(!configFileIn)
  {
    cerr << "Error: File " << argv[1] << " is not valid. Exiting" << endl;
    return 1;
  }

  //File is good
  parseConfig(configFileIn, configVm);

  //Close config when done
  configFileIn.close();

  try
  {
    //Extract all the values to their respective global variables
    trainingSet = configVm["CNF Training Set"].as<string>();
    testSet = configVm["CNF Test Set"].as<string>();
    solutionFile = configVm["Solution File"].as<string>();
    trainingLog = configVm["Training Log File"].as<string>();
    testLog = configVm["Test Log File"].as<string>();
    numRuns = configVm["Number of Runs"].as<int>();
    evalsPerRun = configVm["Number of Evals per Run"].as<int>();
    populationSize = configVm["Population Size"].as<int>();
    offspringPerGeneration = configVm["Offspring per generation"].as<int>();
    bestIndividualsFileName = configVm["Load best from file"].as<string>();
    mutationRate = configVm["Mutation chance"].as<double>();
    parentSelectionTournamentSize = configVm["Parent Selection Tournament Size"].as<int>();
    survivalSelectionTournamentSize = configVm["Survival Selection Tournament Size"].as<int>();
    terminateOnMaxEvals = configVm["Terminate on Max Number of Evals"].as<bool>();
    numGenerationsUntilStagnation = configVm["Number of generations of stagnation until termination"].as<int>();
    terminateOnStagnantAverageFitness = configVm["Terminate on Stagnant Average Fitness"].as<bool>();
    terminateOnStagnantBestFitness = configVm["Terminate on Stagnant Best Fitness"].as<bool>();

    //Load individuals special case
    string filenameLower = bestIndividualsFileName;
    boost::algorithm::to_lower(filenameLower);

    if(filenameLower != "false")
    {
      //Load in some individuals
      bestIndividualsFileIn.open(bestIndividualsFileName);

      if(bestIndividualsFileIn)
      {
        //First line is how many individuals
        bestIndividualsFileIn >> numIndividualsToReadIn;

        //Junk read in to get past run #
        bestIndividualsFileIn >> filenameLower;

        if(numIndividualsToReadIn > populationSize)
        {
          numIndividualsToReadIn = populationSize;
        }

        readInBestIndividuals = true;
      }
    }

    //Self-adaptive mutation special case
    string selfAdaptiveMutationString = configVm["Self-adaptive mutation"].as<string>();
    boost::algorithm::to_lower(selfAdaptiveMutationString);

    if(selfAdaptiveMutationString == "true")
    {
      selfAdaptiveMutation = true;
    }
    else if(selfAdaptiveMutationString == "false")
    {
      selfAdaptiveMutation = false;
    }
    else
    {
      cout << "Error: \"" << selfAdaptiveMutationString << "\" is an invalid choice. Please choose true or false"
           << endl;
      std::exit(1);
    }

    //Seed special case
    string seedString = configVm["Time-initialized seed"].as<string>();
    boost::algorithm::to_lower(seedString);
    if(seedString == "true")
    {
      //Set seed to current time in micros
      std::chrono::high_resolution_clock::duration duration = std::chrono::high_resolution_clock::now().time_since_epoch();
      seed = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    }
    else
    {
      try
      {
        seed = std::stol(seedString);
      }
      catch(std::invalid_argument e)
      {
        cout << "Error: \"" << seedString
             << "\" is not a valid seed. Please only use seeds from logfiles\n"
                 " or a value of \"true\" if you want a time-generated seed" << endl;
      }
    }

    std::srand(seed);

    //r-restart special case
    string restartString = configVm["r-restart r value"].as<string>();
    boost::algorithm::to_lower(restartString);

    if(restartString == "false")
    {
      restartsEnabled = false;
      rRestartValue = populationSize;
    }
    else
    {
      restartsEnabled = true;

      try
      {
        rRestartValue = std::stoi(restartString);
      }
      catch(std::invalid_argument)
      {
        cout << "\"" << restartString << "\" " << "is not a valid integer. Try again" << endl;
        return 1;
      }
    }

    //Enum special case
    genesInitMethods[0] = static_cast<InitializationMethod>(configVm["Luby Initialization"].as<int>());
    genesInitMethods[1] = static_cast<InitializationMethod>(configVm["Rnd-Freq Initialization"].as<int>());
    genesInitMethods[2] = static_cast<InitializationMethod>(configVm["Var-Decay Initialization"].as<int>());
    genesInitMethods[3] = static_cast<InitializationMethod>(configVm["Cla-Decay Initialization"].as<int>());
    genesInitMethods[4] = static_cast<InitializationMethod>(configVm["Rinc Initialization"].as<int>());
    genesInitMethods[5] = static_cast<InitializationMethod>(configVm["GC-Frac Initialization"].as<int>());
    genesInitMethods[6] = static_cast<InitializationMethod>(configVm["RFirst Initialization"].as<int>());
    genesInitMethods[7] = static_cast<InitializationMethod>(configVm["CCMIN-Mode Initialization"].as<int>());
    genesInitMethods[8] = static_cast<InitializationMethod>(configVm["Phase-Saving Initialization"].as<int>());
    
    
    parentSelectionMethod = static_cast<ParentSelectionMethod>(configVm["Parent Selection"].as<int>());
    survivalSelectionMethod = static_cast<SurvivalSelectionMethod>(configVm["Survival Selection"].as<int>());
    survivalStrategy = static_cast<SurvivalStrategy>(configVm["Survival Strategy"].as<int>());
  }
  catch(boost::bad_any_cast& error)
  {
    cerr << error.what() << endl;
    return 1;
  }

  //Open streams
  trainingLogFileOut.open(trainingLog);
  testLogFileOut.open(testLog);
  solutionFileOut.open(solutionFile);
  lastPopulationFileOut.open("seeds/" + std::to_string(seed) + ".txt");

  time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  configString = "\n=== Config ===";
  configString += "\nStarted at: ";
  configString += std::ctime(&time);
  configString += "CNF Training Set: " + trainingSet;
  configString += "\nCNF Test Set: " + testSet;
  configString += "\nSolution File: " + solutionFile;
  configString += "\nTraining Log File: " + trainingLog;
  configString += "\nTest Log File: " + testLog;
  configString += "\nLoaded individuals from file: " + bestIndividualsFileName;
  configString += "\nNumber of Runs: " + std::to_string(numRuns);
  configString += "\nNumber of evals per run: " + std::to_string(evalsPerRun);
  configString += "\nSelf-adaptive mutation: ";
  configString += (selfAdaptiveMutation ? "true" : "false");
  configString += "\nSeed: " + std::to_string(seed);
  configString += "\nr-restart r value: " + (rRestartValue == populationSize ? "false" : std::to_string(rRestartValue));
  configString += "\nPopulation Size: " + std::to_string(populationSize);
  configString += "\nOffspring per generation: " + std::to_string(offspringPerGeneration);
  configString += "\nMutation Chance: " + std::to_string(mutationRate);
  configString += "\nLuby Initialization: " + std::to_string(static_cast<int>(genesInitMethods[0]));
  configString += "\nRnd-Freq Initialization: " + std::to_string(static_cast<int>(genesInitMethods[1]));
  configString += "\nVar-Decay Initialization: " + std::to_string(static_cast<int>(genesInitMethods[2]));
  configString += "\nCla-Decay Initialization: " + std::to_string(static_cast<int>(genesInitMethods[3]));
  configString += "\nRinc Initialization: " + std::to_string(static_cast<int>(genesInitMethods[4]));
  configString += "\nGC-Frac Initialization: " + std::to_string(static_cast<int>(genesInitMethods[5]));
  configString += "\nRFirst Initialization: " + std::to_string(static_cast<int>(genesInitMethods[6]));
  configString += "\nCCMIN-Mode Initialization: " + std::to_string(static_cast<int>(genesInitMethods[7]));
  configString += "\nPhase-Saving Initialization: " + std::to_string(static_cast<int>(genesInitMethods[8]));
  configString += "\nParent Selection: " + std::to_string(static_cast<int>(parentSelectionMethod));
  configString += "\nParent Selection Tournament Size: " + std::to_string(parentSelectionTournamentSize);
  configString += "\nSurvival Strategy: ";
  configString += std::to_string(static_cast<int>(survivalStrategy));
  configString += "\nSurvival Selection: " + std::to_string(static_cast<int>(survivalSelectionMethod));
  configString += "\nSurvival Selection Tournament Size: " + std::to_string(survivalSelectionTournamentSize);
  configString += "\nTerminate on max evals: ";
  configString += (terminateOnMaxEvals ? "true" : "false");
  configString += "\nTerminate on stagnant average fitness: ";
  configString += (terminateOnStagnantAverageFitness ? "true" : "false");
  configString += "\nTerminate on stagnant best fitness: ";
  configString += (terminateOnStagnantBestFitness ? "true" : "false");
  configString += "\n=== End Config ===";

  cout << configString << endl;

  trainingLogFileOut << "Training Result Log" << configString << endl;
  testLogFileOut << "Test Result Log" << configString << endl;

  generationsBetweenTestSetEvals =
      static_cast<int>(std::round(((evalsPerRun - populationSize) / static_cast<double>(offspringPerGeneration))
                                  / TIMES_TO_EVALUATE_AGAINST_TEST_SET));


  //Initialize dynamic memory
  //The population. Stays static at <populationSize> members
  Genome** population = new Genome *[populationSize];

  for(int j = 0; j < populationSize; j++)
  {
    population[j] = nullptr;
  }

  //The population plus their kids
  Genome** kids = new Genome *[offspringPerGeneration];

  //List to hold parents and children together. Useful for survival selection
  Genome** parentsAndKids = new Genome *[populationSize + offspringPerGeneration];

  //Breeding pool for parents
  Genome** breedingPool = new Genome *[2 * offspringPerGeneration];

  //Start runs
  for(int i = 1; i <= numRuns; i++)
  {
    cout << "Run #" << i << endl;
    trainingLogFileOut << "Run #" << i << endl;
    testLogFileOut << "Run #" << i << endl;

    //Set up each run
    initializePopulation(population, populationSize, genesInitMethods, trainingSet);

    bestIndividualsFileIn.close();

    bestPopulationMember = findExtremeIndividual(population, populationSize, true);
    printLineInLog(trainingLogFileOut, population, populationSize);

    //Run until termination
    while(!terminationConditionMet())
    {
      cout << "=============== Generation " << generation << " ===============" << endl;

      if(justRestarted)
      {
        cout << rRestartValue << "-RESTART" << endl;
        justRestarted = false;

        //Initialize mu - r members
        for(int j = 0; j < rRestartValue; j++)
        {
          cout << "ID: " << population[j] -> id << " Fitness: " << population[j] -> trainingFitness << endl;
        }

        initializePopulation(population, populationSize, genesInitMethods, trainingSet);
      }
      //breeding pool will be filled with 2*lambda parents (repeats are possible)
      parentSelection(population, populationSize, breedingPool, 2 * offspringPerGeneration, parentSelectionTournamentSize,
                      parentSelectionMethod);

      recombination(breedingPool, 2 * offspringPerGeneration, kids, offspringPerGeneration, trainingSet);

      //Check on survival strategy
      if(survivalStrategy == SurvivalStrategy::MU_COMMA_LAMBDA)
      {
        //Delete every member of population
        for(int j = 0; j < populationSize; j++)
        {
          if(population[j] != nullptr)
          {
            delete population[j];
            population[j] = nullptr;
          }
        }

        if(offspringPerGeneration < populationSize)
        {
          cout << "Lambda is less than mu, resizing dynamic memory and setting mu to lambda" << endl;
          //User is officially an idiot. lambda < mu but mu individuals are always killed off, mu is effectively
          //equal to lambda now. Resize dynamic memory

          delete[] population;

          populationSize = offspringPerGeneration;
          population = new Genome *[populationSize];

          delete[] parentsAndKids;

          //Re-declare this useless array so we can free it at the end
          parentsAndKids = new Genome *[populationSize + offspringPerGeneration];
        }

        //mu == lambda, so no survival selection needs to happen. Kids always survive
        for(int j = 0; j < offspringPerGeneration; j++)
        {
          population[j] = kids[j];
        }
      }
      else
      {
        //Mush the population array together with the kids array
        for(int j = 0; j < populationSize; j++)
        {
          parentsAndKids[j] = population[j];
        }

        for(int j = 0; j < offspringPerGeneration; j++)
        {
          parentsAndKids[j + populationSize] = kids[j];
        }

        survivalSelection(parentsAndKids, populationSize, offspringPerGeneration, survivalSelectionTournamentSize,
                          survivalSelectionMethod);

        //Put non-null members of parentsAndKids into general population while nulling out parentsAndKids
        int x = 0;
        for(int j = 0; j < populationSize + offspringPerGeneration; j++)
        {
          if(parentsAndKids[j] != nullptr)
          {
            population[x] = parentsAndKids[j];
            parentsAndKids[j] = nullptr;
            x++;
          }
        }
      }

      for(int j = 0; j < offspringPerGeneration; j++)
      {
        kids[j] = nullptr;
      }

      for(int j = 0; j < 2 * offspringPerGeneration; j++)
      {
        breedingPool[j] = nullptr;
      }

      previousAverageFitness = currentAverageFitness;
      currentAverageFitness = findAverageFitness(population, populationSize);

      if(currentAverageFitness == previousAverageFitness)
      {
        generationsAverageStagnant++;
        cout << "Average Fitness has been stagnant for " << generationsAverageStagnant << " generations" << endl;
      }
      else
      {
        generationsAverageStagnant = 0;
      }

      previousBestFitness = bestPopulationMember -> trainingFitness;
      bestPopulationMember = findExtremeIndividual(population, populationSize, true);

      if(bestPopulationMember -> trainingFitness == previousBestFitness)
      {
        generationsBestStagnant++;
        cout << "Best fitness has been stagnant for " << generationsBestStagnant << " generations" << endl;
      }
      else
      {
        generationsBestStagnant = 0;
      }

      //Test to see if we should evaluate the best member against the test set
      if(generationsBetweenTestSetEvals == 0 || generation % generationsBetweenTestSetEvals == 0)
      {
        cout << "Evaluating best member against test set" << endl;

        bestPopulationMember -> testFitness = evaluateFitness(testSet, bestPopulationMember);

        if(bestPopulationMember -> testFitness > bestOverallTestFitness)
        {
          bestOverallTestFitness = bestPopulationMember -> testFitness;
          absoluteBestParamString = bestPopulationMember -> getParamString();
        }

        testLogFileOut << numEvals << "\t" << bestPopulationMember -> testFitness << endl;
        cout << "ID: " << bestPopulationMember -> id << " Test Fitness: "
             << bestPopulationMember -> testFitness << endl;
      }

      printLineInLog(trainingLogFileOut, population, populationSize);
      generation++;

      //Determine if restart is necessary
      if(restartsEnabled)
      {
        //If we have converged and we have enough evals left to finish at least one generation (init + children)
        if(populationHasConverged() && evalsPerRun - numEvals >= (populationSize - rRestartValue + offspringPerGeneration))
        {
          justRestarted = true;
          readInBestIndividuals = false;
          Genome** bestMembers = new Genome*[rRestartValue];

          quicksort(population, 0, populationSize - 1);

          //Copy the r best members
          for(int j = 0; j < rRestartValue; j++)
          {
            bestMembers[j] = population[j];
          }

          //Kill entire population
          for(int j = rRestartValue; j < populationSize; j++)
          {
            if(population[j] != nullptr)
            {
              delete population[j];
              population[j] = nullptr;
            }
          }

          //Pre-load population with r best members
          for(int j = 0; j < rRestartValue; j++)
          {
            population[j] = bestMembers[j];
          }

          delete []  bestMembers;
        }
      }
    }

    //Output last population from each run
    lastPopulationFileOut << populationSize << endl;
    lastPopulationFileOut << "Run " << i << endl;

    //Clean up each run
    for(int j = 0; j < populationSize; j++)
    {
      if(population[j] != nullptr)
      {
        lastPopulationFileOut << (population[j] -> luby ? "true" : "false") << "\t"
                              << population[j] -> rnd_freq << "\t"
                              << population[j] -> var_decay << "\t"
                              << population[j] -> cla_decay << "\t"
                              << (population[j] -> rinc / std::numeric_limits<double>::max()) << "\t"
                              << (population[j] -> gc_frac / std::numeric_limits<double>::max()) << "\t"
                              << population[j] -> rfirst << "\t"
                              << population[j] -> ccmin_mode <<  "\t"
                              << population[j] -> phase_saving << "\t"
                              << population[j] -> trainingFitness << endl;
        delete population[j];
        population[j] = nullptr;
      }
    }

    lastPopulationFileOut << endl;

    // Reset pertinent variables
    totalIndividualsProduced = 0;
    numEvals = 0;
    generation = 1;
    bestPopulationMember = nullptr;
    worstPopulationMember = nullptr;
    generationsAverageStagnant = 0;
    generationsBestStagnant = 0;
    justRestarted = false;

    //Throw newlines into the logs
    trainingLogFileOut << endl;
    testLogFileOut << endl;
  }

  //Write best solution to solution file
  solutionFileOut << "./solvers/minisat/minisat -cpu-lim=5 " << absoluteBestParamString << endl;
  time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  trainingLogFileOut << "Ended at " << ctime(&time);
  testLogFileOut << "Ended at " << ctime(&time);

  //Release dynamic memory
  delete[] kids;
  delete[] population;
  delete[] breedingPool;
  delete[] parentsAndKids;

  delete rng;
  delete gen;

  //Close logs
  trainingLogFileOut.close();
  testLogFileOut.close();
  solutionFileOut.close();
  lastPopulationFileOut.close();

  return 0;
}

void parseConfig(ifstream& configFileIn, boost::program_options::variables_map& configVm)
{
  try
  {
    boost::program_options::options_description config{"General"};

    config.add_options()
        ("CNF Training Set", boost::program_options::value<string>(), "Directory of training set")
        ("CNF Test Set", boost::program_options::value<string>(), "Directory of test set")
        ("Solution File", boost::program_options::value<string>(), "Filepath of solution file")
        ("Training Log File", boost::program_options::value<string>(), "Filepath of training log file")
        ("Test Log File", boost::program_options::value<string>(), "Filepath of test log file")
        ("Number of Runs", boost::program_options::value<int>(), "Number of runs to do per experiment")
        ("Number of Evals per Run", boost::program_options::value<int>(), "Max number of fitness evaluations per run")
        ("Self-adaptive mutation", boost::program_options::value<string>())
        ("Load best from file", boost::program_options::value<string>())
        ("Time-initialized seed", boost::program_options::value<string>())
        ("r-restart r value", boost::program_options::value<string>())
        ("Population Size", boost::program_options::value<int>(), "Population size")
        ("Offspring per generation", boost::program_options::value<int>())
        ("Mutation chance", boost::program_options::value<double>())
        ("Luby Initialization", boost::program_options::value<int>())
        ("Rnd-Freq Initialization", boost::program_options::value<int>())
        ("Var-Decay Initialization", boost::program_options::value<int>())
        ("Cla-Decay Initialization", boost::program_options::value<int>())
        ("Rinc Initialization", boost::program_options::value<int>())
        ("GC-Frac Initialization", boost::program_options::value<int>())
        ("RFirst Initialization", boost::program_options::value<int>())
        ("CCMIN-Mode Initialization", boost::program_options::value<int>())
        ("Phase-Saving Initialization", boost::program_options::value<int>())
        ("Parent Selection", boost::program_options::value<int>())
        ("Parent Selection Tournament Size", boost::program_options::value<int>())
        ("Survival Strategy", boost::program_options::value<int>())
        ("Survival Selection", boost::program_options::value<int>())
        ("Survival Selection Tournament Size", boost::program_options::value<int>())
        ("Terminate on Max Number of Evals", boost::program_options::value<bool>())
        ("Number of generations of stagnation until termination", boost::program_options::value<int>())
        ("Terminate on Stagnant Average Fitness", boost::program_options::value<bool>())
        ("Terminate on Stagnant Best Fitness", boost::program_options::value<bool>());

    //Parse the file and store the config variables
    boost::program_options::store(boost::program_options::parse_config_file(configFileIn, config), configVm);
    boost::program_options::notify(configVm);
  }
  catch(boost::program_options::error& error)
  {
    cerr << error.what() << endl;
    std::exit(1);
  }
}

void uniformRandomInitialization(Genome* individual, int geneIndex)
{
  double reallySmallValue = .0000000001;
  //rand() / RAND_MAX -> gives a double in the range [0, 1]

  if(selfAdaptiveMutation)
  {
    for(int i = 0; i < 9; i++)
    {
      individual -> sigma[i] = static_cast<double>(std::rand()) / RAND_MAX;
    }
  }

  switch(geneIndex)
  {
    case 0:
      //luby or no-luby (in set {0, 1})
      individual->luby = std::rand() % 2 == 1;
      break;
    case 1:
      //rnd-freq -> [0, 1]
      individual->rnd_freq = static_cast<double>(std::rand()) / RAND_MAX;
      break;
    case 2:
      //var-decay -> (0, 1)
      individual->var_decay = static_cast<double>(std::rand()) / RAND_MAX;
      //Make the range exclusive
      if(individual->var_decay == 0)
      {
        individual->var_decay = reallySmallValue;
      }
      else if(individual->var_decay == 1)
      {
        individual->var_decay = 1 - reallySmallValue;
      }
      break;
    case 3:
      //cla-decay -> (0, 1)
      individual->cla_decay = static_cast<double>(std::rand()) / RAND_MAX;

      //Make the ranges exclusive
      if(individual->cla_decay == 0)
      {
        individual->cla_decay = reallySmallValue;
      }
      else if(individual->cla_decay == 1)
      {
        individual->cla_decay = 1 - reallySmallValue;
      }
      break;
    case 4:
      //rinc -> (1, inf)
      individual->rinc = (std::numeric_limits<double>::max() - 1 - reallySmallValue) *
                         (static_cast<double>(std::rand()) / RAND_MAX) + 1 + reallySmallValue;
      break;
    case 5:
      //gc-frac -> (0, inf)
      individual->gc_frac = (std::numeric_limits<double>::max() - reallySmallValue) *
                            (static_cast<double>(std::rand()) / RAND_MAX) + reallySmallValue;
      break;
    case 6:
      //rfirst -> [1, INT_MAX]
      individual->rfirst = static_cast<int>((std::numeric_limits<int>::max() - 1) *
          (static_cast<double>(std::rand()) / RAND_MAX)) + 1;
      break;
    case 7:
      //ccmin-mode -> {0, 1, 2}
      individual->ccmin_mode = std::rand() % 3;
      break;
    case 8:
      //phase-saving -> {0, 1, 2}
      individual->phase_saving = std::rand() % 3;
      break;
    default:
      cout << "Error in uniform init!" << endl;
      std::exit(1);
  }
}

void biasedInitialization(Genome* individual, int geneIndex, InitializationMethod biasValue)
{
  double influence = 1, min = 0, max = 1, bias, value, rand1, rand2;

  rand1 = static_cast<double>(std::rand()) / RAND_MAX;
  rand2 = static_cast<double>(std::rand()) / RAND_MAX;

  if(biasValue == InitializationMethod::BIASED_TOWARDS_SMALLER_VALUES)
  {
    bias = 1.0/4;
  }
  else if(biasValue == InitializationMethod::BIASED_TOWARDS_LARGER_VALUES)
  {
    bias = 3.0/4;
  }
  else
  {
    cout << "Logic error in biased init!" << endl;
    std::exit(1);
  }

  value = (rand1 * (max - min) + min) * (1 - (rand2 * influence) + bias * (rand2 * influence));

  if(selfAdaptiveMutation)
  {
    for(int i = 0; i < 9; i++)
    {
      rand1 = static_cast<double>(std::rand()) / RAND_MAX;
      rand2 = static_cast<double>(std::rand()) / RAND_MAX;

      individual->sigma[i] = (rand1 * (max - min) + min) * (1 - (rand2 * influence) + bias * (rand2 * influence));
    }
  }

  if(value > 1)
  {
    value = 1;
  }
  else if(value < 0)
  {
    value = 0;
  }
  double reallySmallValue = .0000000001;
  //rand() / RAND_MAX -> gives a double in the range [0, 1]

  switch(geneIndex)
  {
    case 0:
      //luby or no-luby (in set {0, 1})
      individual->luby = (std::round(value) == 1);
      break;
    case 1:
      //rnd-freq -> [0, 1]
      individual->rnd_freq = value;
      break;
    case 2:
      //var-decay -> (0, 1)
      individual->var_decay = value;
      //Make the range exclusive
      if(individual->var_decay == 0)
      {
        individual->var_decay = reallySmallValue;
      }
      else if(individual->var_decay == 1)
      {
        individual->var_decay = 1 - reallySmallValue;
      }
      break;
    case 3:
      //cla-decay -> (0, 1)
      individual->cla_decay = value;

      //Make the ranges exclusive
      if(individual->cla_decay == 0)
      {
        individual->cla_decay = reallySmallValue;
      }
      else if(individual->cla_decay == 1)
      {
        individual->cla_decay = 1 - reallySmallValue;
      }
      break;
    case 4:
      //rinc -> (1, inf)
      individual->rinc = (std::numeric_limits<double>::max() - 1 - reallySmallValue) * value + 1 + reallySmallValue;
      break;
    case 5:
      //gc-frac -> (0, inf)
      individual->gc_frac = (std::numeric_limits<double>::max() - reallySmallValue) * value + reallySmallValue;
      break;
    case 6:
      //rfirst -> [1, INT_MAX]
      individual->rfirst = static_cast<int>((std::numeric_limits<int>::max() - 1) * value) + 1;
      break;
    case 7:
      //ccmin-mode -> {0, 1, 2}
      individual->ccmin_mode = static_cast<int>(100 * value) % 3;
      break;
    case 8:
      //phase-saving -> {0, 1, 2}
      individual->phase_saving = static_cast<int>(100 * value) % 3;
      break;
    default:
      cout << "Error in uniform init!" << endl;
      std::exit(1);
  }
}

Genome* readInIndividual(ifstream& fileIn)
{
  Genome* individual = new Genome();
  string tmp;

  fileIn >> tmp;

  individual -> luby = (tmp == "true");

  fileIn >> individual -> rnd_freq;
  fileIn >> individual -> var_decay;
  fileIn >> individual -> cla_decay;
  fileIn >> individual -> rinc;
  fileIn >> individual -> gc_frac;
  fileIn >> individual -> rfirst;
  fileIn >> individual -> ccmin_mode;
  fileIn >> individual -> phase_saving;
  fileIn >> individual -> trainingFitness;
}

void initializePopulation(Genome** population, int populationSize, InitializationMethod initMethods[9],
                          string trainingSet)
{
  if(readInBestIndividuals)
  {
    int individualsRead, index = 0;
    while(individualsRead < numIndividualsToReadIn)
    {
      if(population[index] == nullptr)
      {
        population[index] = readInIndividual(bestIndividualsFileIn);
        population[index] -> id = totalIndividualsProduced;
        totalIndividualsProduced++;
        index++;
      }
    }
  }

  for(int i = 0; i < populationSize; i++)
  {
    //Member needs to be initialized
    if(population[i] == nullptr)
    {
      //Create the memory for the genome
      population[i] = new Genome();

      totalIndividualsProduced++;

      //Assign it a unique id
      population[i]->id = totalIndividualsProduced;

      for(int j = 0; j < 9; j++)
      {
        switch(initMethods[j])
        {
          case InitializationMethod::UNIFORM_RANDOM:
            uniformRandomInitialization(population[i], j);
            break;
          case InitializationMethod::BIASED_TOWARDS_LARGER_VALUES:
          case InitializationMethod::BIASED_TOWARDS_SMALLER_VALUES:
            //Fall-through on purpose
            biasedInitialization(population[i], j, initMethods[j]);
            break;
        }
      }

      //Make sure params are in the right range
      checkParamRanges(population[i]);

      //Evaluate its fitness
      population[i]->trainingFitness = evaluateFitness(trainingSet, population[i]);

      cout << "ID: " << population[i]->id << " Fitness: " << population[i]->trainingFitness << endl;

      //Keep track of evals
      numEvals++;
    }
  }
}
double evaluateFitness(string inputDirectory, Genome* genome)
{
    int exit;
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> duration;

    string command = "./solvers/minisat/minisat -cpu-lim=5 " + genome->getParamString() + " " + inputDirectory +
                     " >> /dev/null";

    start = std::chrono::high_resolution_clock::now();
    exit = system(command.c_str());
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    if(exit != 0)
    {
        cout << "Non-zero exit code!" << endl;
        return std::numeric_limits<double>::lowest();
    }

    return 100 - duration.count();
}
void fitnessProportionalSelection(Genome** population, const int populationSize, Genome** selectedIndividuals,
                                  const int numIndividualsToChoose)
{
  //Find worst population member
  worstPopulationMember = findExtremeIndividual(population, populationSize, false);

  double* aArray = new double[populationSize];
  double totalFitness = 0, runningProbabilitySum = 0;
  int iValue = 0, currentMember = 0;

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

    //Use this running total to generate the a array
    aArray[i] = runningProbabilitySum;
  }

  //rand / RAND_MAX -> [0, 1]
  double rValue = (1 / static_cast<float>(numIndividualsToChoose)) * static_cast<float>(std::rand()) / RAND_MAX;
  while(currentMember < numIndividualsToChoose)
  {
    while(rValue <= aArray[iValue] && !population[iValue] -> chosen)
    {
      selectedIndividuals[currentMember] = population[iValue];
      selectedIndividuals[currentMember] -> chosen = true;
      cout << "ID: " << selectedIndividuals[currentMember]->id << " Fitness: "
           << selectedIndividuals[currentMember]->trainingFitness << endl;
      rValue += (1 / static_cast<double>(numIndividualsToChoose));

      currentMember++;
    }
    iValue++;
  }

  //Clean up dynamic memory
  delete [] aArray;
}
void tournamentSelection(Genome** population, const int populationSize, Genome** tournamentWinners,
                         const int numWinnersToChoose, int tournamentSize, bool withReplacement)
{
  int parentsFound = 0, tournamentMembersFound = 0, parentChoice, indexOfTournamentWinner;
  double maxFitness;
  Genome** selectionTournament = new Genome*[tournamentSize];

  //Find 2*lambda parents and throw them into the breeding pool
  while(parentsFound != numWinnersToChoose)
  {
    //Reset number of members found
    tournamentMembersFound = 0;

    cout << "=== Finding tournament attendees ===" << endl;
    //Randomly pull <tournamentSize> members from the population and put them in a tournament
    while(tournamentMembersFound != tournamentSize)
    {
      //Choose a random member of the population
      parentChoice = std::rand() % populationSize;

      //If they're not already in the tournament, throw them in
      if(population[parentChoice] != nullptr && !population[parentChoice]->enteredInTournament)
      {
        selectionTournament[tournamentMembersFound] = population[parentChoice];
        tournamentMembersFound++;
        cout << "ID: " << population[parentChoice]->id << " Fitness: "
             << population[parentChoice]->trainingFitness << endl;
        population[parentChoice]->enteredInTournament = true;
      }
    }

    //Reset pertinent vars
    indexOfTournamentWinner = 0;
    maxFitness = std::numeric_limits<double>::lowest();
    tournamentMembersFound = 0;

    //Find the winner of this tournament
    for(int i = 0; i < tournamentSize; i++)
    {
      //Signal that they're no longer entered in the tournament - ie they can be picked again
      selectionTournament[i]->enteredInTournament = false;
      if(selectionTournament[i]->trainingFitness > maxFitness)
      {
        maxFitness = selectionTournament[i]->trainingFitness;
        indexOfTournamentWinner = i;
      }
    }

    cout << "Winner: " << selectionTournament[indexOfTournamentWinner]->id << " Fitness: "
         << selectionTournament[indexOfTournamentWinner]->trainingFitness << endl;

    //Useful for killing off non-chosen individuals during survival selection
    selectionTournament[indexOfTournamentWinner] -> chosen = true;

    if(!withReplacement)
    {
      //Signal that the winner can no longer be picked
      selectionTournament[indexOfTournamentWinner] -> enteredInTournament = true;
    }

    //Place the winner in the breeding pool
    tournamentWinners[parentsFound] = selectionTournament[indexOfTournamentWinner];
    parentsFound++;
  }
  parentsFound = 0;
  delete [] selectionTournament;

  if(!withReplacement)
  {
    //Clear the tournament flags for all winners so it doesn't mess something up later
    for(int i = 0; i < numWinnersToChoose; i++)
    {
      tournamentWinners[i] -> enteredInTournament = false;
    }
  }
}
void parentSelection(Genome** population, const int populationSize, Genome** breedingPool,
                     const int parentSelectionPoolSize, const int parentSelectionTournamentSize,
                     ParentSelectionMethod parentSelectionMethod)
{
  switch(parentSelectionMethod)
  {
    case ParentSelectionMethod::UNIFORM_RANDOM:
      cout << "=============== Uniform-Random Parent Selection ===============" << endl;
      for(int i = 0; i < parentSelectionPoolSize; i++)
      {
        breedingPool[i] = population[std::rand() % populationSize];
        cout << "ID " << breedingPool[i] -> id << " Fitness: " << breedingPool[i] -> trainingFitness << endl;
      }
      break;
    case ParentSelectionMethod::FITNESS_PROPORTIONAL:
      cout << "=============== Fitness-Proportional Parent Selection ===============" << endl;
      fitnessProportionalSelection(population, populationSize, breedingPool, parentSelectionPoolSize);
      break;
    case ParentSelectionMethod::TOURNAMENT_SELECTION_WITH_REPLACEMENT:
      cout << "=============== Fitness-Proportional Parent Selection with replacement ===============" << endl;
      tournamentSelection(population, populationSize, breedingPool, parentSelectionTournamentSize,
                          parentSelectionPoolSize, true);
      break;
    default:
      cerr << "Error: Parent Selection Method of " << static_cast<int>(parentSelectionMethod) << " is invalid. Exiting."
           << endl;
      std::exit(1);
  }
}
void recombination(Genome** breedingPool, const int breedingPoolSize, Genome** kids, const int numKids,
                   string trainingSet)
{
  cout << "=============== Recombination & Mutation ===============" << endl;
  //Take the parents, pair them randomly and perform crossover
  Genome *parents[2];

  parents[0] = parents[1] = nullptr;
  int parentsIndex = 0, index;
  for(int i = 0; i < numKids; i++)
  {
    //Choose 2 parents
    while(!(parents[0] != nullptr && parents[1] != nullptr))
    {
      index = std::rand() % (breedingPoolSize);

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
    mutateIndividual(kids[i], mutationRate, rng, gen);

    //Evaluate fitness
    kids[i]->trainingFitness = evaluateFitness(trainingSet, kids[i]);
    cout << "Individual " << kids[i]->id << " (child of " << parents[0]->id << " and " << parents[1]->id
         << ") Fitness: " << kids[i]->trainingFitness << endl;

    numEvals++;
    parents[0] = parents[1] = nullptr;
  }
}

Genome* crossover(const Genome* parent1, const Genome* parent2)
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

  //make sure everything is good
  checkParamRanges(child);
  return child;
}
void mutateIndividual(Genome* individual, double mutationRate, std::normal_distribution<double> *rng, 
                      std::default_random_engine *gen)
{
  std::normal_distribution<double> normalZeroOne(0, 1);

  double mutationChance[9];
  double mutationValues[9];
  
  double currentFraction, reallySmallValue = .0000000001, tau = 1 / std::sqrt(populationSize);

  //9 uniformly-distributed numbers in range [0, 1]
  for(int i = 0; i < 9; i++)
  {
    mutationChance[i] = static_cast<double>(std::rand()) / RAND_MAX;

    if(selfAdaptiveMutation)
    {
      mutationValues[i] = individual -> sigma[i] * normalZeroOne(*gen);
      individual -> sigma[i] *= static_cast<double>(std::exp(tau * normalZeroOne(*gen)));
    }
    else
    {
      mutationValues[i] = rng->operator()(*gen);

      //Cap it to range [-1, 1]
      if(mutationValues[i] > 1)
      {
        mutationValues[i] = 1;
      }
      else if(mutationValues[i] < -1)
      {
        mutationValues[i] = -1;
      }
    }
  }

  //If mutation occurs, randomize their luby value
  individual->luby = mutationChance[0] <= mutationRate ? std::round(mutationValues[0]) == 1 : individual->luby;

  //If mutation occurs, add a normally distributed value to rnd_freq
  if(mutationChance[1] <= mutationRate)
  {
    individual->rnd_freq += mutationValues[1];

    individual->rnd_freq = individual->rnd_freq < 0 ? 0 : individual->rnd_freq;
    individual->rnd_freq = individual->rnd_freq > 1 ? 1 : individual->rnd_freq;
  }

  //If mutation occurs, add a normally distributed value to var_decay
  if(mutationChance[2] <= mutationRate)
  {
    individual->var_decay += mutationValues[2];

    individual->var_decay = individual->var_decay <= 0 ? reallySmallValue : individual->var_decay;
    individual->var_decay = individual->var_decay >= 1 ? 1 - reallySmallValue : individual->var_decay;
  }

  //If mutation occurs, add a normally distributed value to cla_decay
  if(mutationChance[3] <= mutationRate)
  {
    individual->cla_decay += mutationValues[3];

    individual->cla_decay = individual->cla_decay <= 0 ? reallySmallValue : individual->cla_decay;
    individual->cla_decay = individual->cla_decay >= 1 ? 1 - reallySmallValue : individual->cla_decay;
  }

  //If mutation occurs, add a normally distributed value to rinc
  if(mutationChance[4] <= mutationRate)
  {
    //Obtain the current fraction of max that rinc has achieved
    currentFraction = individual -> rinc / std::numeric_limits<double>::max();

    currentFraction += mutationValues[4];

    //If adding x will overflow a double
    if(currentFraction >= 1)
    {
      individual->rinc = (1- reallySmallValue) * std::numeric_limits<double>::max();
    }
    else if(currentFraction <= (1.0 / std::numeric_limits<double>::max()))
    {
      //Fraction goes below the exclusive lower bound of 1
      individual->rinc = 1 + reallySmallValue;
    }
    else
    {
      //Guaranteed fraction in range (1/DBL_MAX, 1]
      individual->rinc = currentFraction * std::numeric_limits<double>::max();
    }
  }

  //If mutation occurs, add a normally distributed value to gc_frac
  if(mutationChance[5] <= mutationRate)
  {
    currentFraction = individual -> gc_frac / std::numeric_limits<double>::max();
    currentFraction += mutationValues[5];

    //If adding x will overflow a double
    if(currentFraction >= 1)
    {
      individual->gc_frac = (1 - reallySmallValue) * std::numeric_limits<double>::max();
    }
    else if(currentFraction <= 0)
    {
      //Fraction breaks exclusive lower bound of 0
      individual->gc_frac = reallySmallValue;
    }
    else
    {
      individual->gc_frac = currentFraction * std::numeric_limits<double>::max();
    }
  }

  //If mutation occurs, add a normally distributed value to rfirst
  if(mutationChance[6] <= mutationRate)
  {
    currentFraction = (static_cast<double>(individual -> rfirst) / std::numeric_limits<int>::max());

    currentFraction += mutationValues[6];

    //If adding y overflows an int
    if(currentFraction > 1)
    {
      individual->rfirst = std::numeric_limits<int>::max();
    }
    else if(currentFraction < (1.0 / std::numeric_limits<int>::max()))
    {
      //Fraction would have given less than 1
      individual->rfirst = 1;
    }
    else
    {
      individual->rfirst = static_cast<int>(currentFraction * std::numeric_limits<int>::max());
    }
  }

  //If mutation occurs, randomize their ccmin_mode value
  individual->ccmin_mode = mutationChance[7] <= mutationRate ? std::rand() % 3 : individual->ccmin_mode;

  //If mutation occurs, randomize their phase_saving value
  individual->phase_saving = mutationChance[8] <= mutationRate ? std::rand() % 3 : individual->phase_saving;

  //Check to make sure everything is gucci
  checkParamRanges(individual);
}
void survivalSelection(Genome** entirePopulation, const int populationSize, const int numKids,
                       const int survivalSelectionTournamentSize, SurvivalSelectionMethod survivalSelectionMethod)
{
  Genome** survivors = new Genome*[populationSize];
  int chosenIndex = 0;
  switch(survivalSelectionMethod)
  {
    case SurvivalSelectionMethod::FITNESS_PROPORTIONAL:
      cout << "=============== Fitness-Proportional Survival Selection ===============" << endl;
      fitnessProportionalSelection(entirePopulation, populationSize + numKids, survivors, populationSize);
      break;
    case SurvivalSelectionMethod::UNIFORM_RANDOM:
      cout << "=============== Uniform-Random Survival Selection ===============" << endl;
      for(int i = 0; i < populationSize; i++)
      {
        do
        {
          //Select popSize individuals to be 'chosen'. All others will be killed
          chosenIndex = rand() % (numKids + populationSize);
        }
        while(entirePopulation[chosenIndex] -> chosen);

        entirePopulation[chosenIndex] -> chosen = true;
      }
      break;
    case SurvivalSelectionMethod::TOURNAMENT_SELECTION_WITHOUT_REPLACEMENT:
      cout << "=============== Tournament Survival Selection w/o Replacement ===============" << endl;
      tournamentSelection(entirePopulation, populationSize + numKids, survivors, populationSize,
                          survivalSelectionTournamentSize, false);
      break;
    case SurvivalSelectionMethod::TRUNCATION:
      cout << "=============== Truncation Survival Selection ===============" << endl;
      quicksort(entirePopulation, 0, populationSize + numKids - 1);

      for(int i = 0; i < populationSize; i++)
      {
        entirePopulation[i] -> chosen = true;
      }
      break;
    default:
      cerr << "Error: Survival Selection Method of " << static_cast<int>(survivalSelectionMethod)
           << " is invalid. Exiting." << endl;
      std::exit(1);
  }

  for(int i = 0; i < populationSize + numKids; i++)
  {
    if(entirePopulation[i] != nullptr)
    {
      //If they aren't marked as chosen, they die
      if(!entirePopulation[i] -> chosen)
      {
        cout << "Killing individual " << entirePopulation[i] -> id << endl;
        delete entirePopulation[i];
        entirePopulation[i] = nullptr;
      }
      else
      {
        //Reset their chosen value
        entirePopulation[i] -> chosen = false;
      }
    }
  }

  delete [] survivors;
}
bool terminationConditionMet()
{
  return numEvals >= evalsPerRun;
}

void randomGenome(Genome *genome)
{
  double reallySmallValue = .0000000001;    
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
    genome->var_decay = reallySmallValue;
  }
  else if(genome->var_decay == 1)
  {
    genome->var_decay = 1 - reallySmallValue;
  }

  //cla-decay -> (0, 1)
  genome->cla_decay = static_cast<double>(std::rand()) / RAND_MAX;

  //Make the ranges exclusive
  if(genome->cla_decay == 0)
  {
    genome->cla_decay = reallySmallValue;
  }
  else if(genome->cla_decay == 1)
  {
    genome->cla_decay = 1 - reallySmallValue;
  }
    
  //rinc -> (1, inf)
  genome->rinc = (std::numeric_limits<double>::max() - 1 - reallySmallValue) *
                         (static_cast<double>(std::rand()) / RAND_MAX) + 1 + reallySmallValue;
  //gc-frac -> (0, inf)
  genome->gc_frac = (std::numeric_limits<double>::max() - reallySmallValue) *
                            (static_cast<double>(std::rand()) / RAND_MAX) + reallySmallValue;

  //rfirst -> [1, INT_MAX]
  genome->rfirst = (std::numeric_limits<int>::max() - 1) * (static_cast<double>(std::rand()) / RAND_MAX) + 1;

  //ccmin-mode -> {0, 1, 2}
  genome->ccmin_mode = std::rand() % 3;

  //phase-saving -> {0, 1, 2}
  genome->phase_saving = std::rand() % 3;

  checkParamRanges(genome);
}

void checkParamRanges(const Genome* genome)
{
    //Assert luby is true or false
    assert(genome->luby || !genome->luby);

    //Assert rnd_freq is in [0, 1]
    assert(genome->rnd_freq >= 0);
    assert(genome->rnd_freq <= 1);

    //Assert var_decay is in (0, 1)
    assert(genome->var_decay > 0);
    assert(genome->var_decay < 1);

    //Assert cla_decay is in (0, 1)
    assert(genome->cla_decay > 0);
    assert(genome->cla_decay < 1);

    //Assert rinc is in (1, inf)
    assert(genome->rinc > 1);
    assert(genome->rinc < std::numeric_limits<double>::infinity());

    //Assert gc_frac is in (0, inf)
    assert(genome->gc_frac > 0);
    assert(genome->gc_frac < std::numeric_limits<double>::infinity());

    //Assert rfirst is in [1, INT_MAX]
    assert(genome->rfirst >= 1);
    assert(genome->rfirst <= std::numeric_limits<int>::max());

    //Assert ccmin-mode is in {0, 1, 2}
    assert(genome->ccmin_mode == 0 || genome->ccmin_mode == 1 || genome->ccmin_mode == 2);

    //Assert ccmin-mode is in {0, 1, 2}
    assert(genome->phase_saving == 0 || genome->phase_saving == 1 || genome->phase_saving == 2);
}

void quicksort(Genome** population, int startIndex, int endIndex)
{
  ///WARNING: THIS QUICKSORT SORTS BACKWARDS ON PURPOSE. LARGEST IN FRONT, SMALLEST IN BACK
  int pivot;

  if(startIndex < endIndex)
  {
    pivot = partition(population, startIndex, endIndex);
    quicksort(population, startIndex, pivot - 1);
    quicksort(population, pivot + 1, endIndex);
  }
}

int partition(Genome** population, int startIndex, int endIndex)
{
  //Choose the pivot point as the end
  int pivotIndex = endIndex;
  Genome* temp;
  //Declare counters i,j and put them at the start
  int i,j;

  i = j = startIndex;

  for(int x = startIndex; x < endIndex; x++)
  {
    if(population[i] -> trainingFitness > population[pivotIndex] -> trainingFitness)
    {
      //Swap i and j, increment j
      temp = population[i];
      population[i] = population[j];
      population[j] = temp;
      j++;
    }
    i++;
  }

  //Loop is done, swap j and pivot
  temp = population[pivotIndex];
  population[pivotIndex] = population[j];
  population[j] = temp;

  return j;
}

Genome* findExtremeIndividual(Genome** population, const int populationSize, const bool best)
{
  Genome* extremeIndividual = nullptr;
  double extremeFitness  = (best ? std::numeric_limits<double>::lowest() : std::numeric_limits<double>::max());

  for(int i = 0; i < populationSize; i++)
  {
    if(best)
    {
      if(population[i] -> trainingFitness > extremeFitness)
      {
        extremeFitness = population[i] -> trainingFitness;
        extremeIndividual = population[i];
        absoluteBestParamString = extremeIndividual -> getParamString();
      }
    }
    else
    {
      if(population[i] -> trainingFitness < extremeFitness)
      {
        extremeFitness = population[i] -> trainingFitness;
        extremeIndividual = population[i];
      }
    }
  }

  return extremeIndividual;
}
double findAverageFitness(Genome** population, const int populationSize)
{
  double averageFitness = 0;
  for(int i = 0; i < populationSize; i++)
  {
    averageFitness += population[i] -> trainingFitness;
  }

  return (averageFitness / populationSize);
}

void printLineInLog(ofstream& logFileOut, Genome** population, const int populationSize)
{
  logFileOut << numEvals << "\t" << findAverageFitness(population, populationSize) << "\t"
             << bestPopulationMember -> trainingFitness << endl;
  cout << numEvals << "\t" << findAverageFitness(population, populationSize) << "\t"
       << bestPopulationMember -> trainingFitness << endl;
}

bool populationHasConverged()
{
  if(terminateOnStagnantAverageFitness)
  {
    if(terminateOnStagnantBestFitness)
    {
      return generationsAverageStagnant >= numGenerationsUntilStagnation ||
             generationsBestStagnant >= numGenerationsUntilStagnation;
    }
    else
    {
      return generationsAverageStagnant >= numGenerationsUntilStagnation;
    }
  }
  else
  {
    return generationsBestStagnant >= numGenerationsUntilStagnation;
  }
}