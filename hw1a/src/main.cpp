#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <climits>
#include "Genome.h"

using std::string;
using std::cout;
using std::endl;

float evaluateFitness(string inputDirectory, const Genome& genes);
void randomGenome(Genome& genes);
const int NUM_EA_PARAMS = 6;
int main(int argc, char* argv[])
{
  string eaConfig[NUM_EA_PARAMS];
  int count = 0;
  float maxFitness = -FLT_MAX, currentFitness = 0;
  string line, key, value;
  string logFile, solutionFile, inputDirectory, bestParamString;
  std::ofstream logFileOut, solutionFileOut;
  int numRuns, evalsPerRun;
  long seed;
  if(argc != 2)
  {
    //Incorrect number of CLI arguments
    cout << "Usage: " << argv[0] << " <configFile>" << endl;
    std::exit(1);
  }

  //A single CLI argument was passed in - hopefully it's our config file
  std::ifstream inputFileStream(argv[1]);

  if(inputFileStream)
  {
    //File exists - time to parse it

    cout << "=============== Parameters ===================" << endl;
    while(getline(inputFileStream, line))
    {
        //Ignore lines starting with '#'
        if(line.size() != 0 && line.at(0) != '#')
        {
            std::istringstream lineStringStream(line);

            //Skip the text up until the '='
            if (getline(lineStringStream, key, '='))
            {
                if (getline(lineStringStream, value))
                {
                    //Take the value and throw it into the config array
                    eaConfig[count] = value;
                    count++;
                    cout << key << ": " << value << endl;
                }
            }
        }
    }
  }
  else
  {
    cout << "Error: File \"" << argv[1] << "\" does not exist" << endl;
    cout << "Usage: " << argv[0] << " <configFile>" << endl;
    exit(1);
  }

  //Parse config inputs to their own variables
  for(int i = 0; i < NUM_EA_PARAMS; i++)
  {
    switch(i)
    {
      //CNF Input Directory
      case 0:
        inputDirectory = eaConfig[i];
        break;
      //Solution File
      case 1:
        solutionFile = eaConfig[i];
        solutionFileOut.open(solutionFile);
        break;
      //Log File
      case 2:
        logFile = eaConfig[i];
        logFileOut.open(logFile);
        break;
      //Number of runs
      case 3:
        try
        {
          numRuns = std::stoi(eaConfig[i]);
        }
        catch(std::invalid_argument e)
        {
          cout << "Error: \"" << eaConfig[i] << "\" is not a valid number" << endl;
          exit(1);
        }
        break;
      //Number of evals per run
      case 4:
        try
        {
          evalsPerRun = std::stoi(eaConfig[i]);
        }
        catch(std::invalid_argument e)
        {
          cout << "Error: \"" << eaConfig[i] << "\" is not a valid number" << endl;
          exit(1);
        }
        break;
      //Seed
      case 5:
        if(eaConfig[i] == "True" || eaConfig[i] == "true")
        {
          //Set seed to current time in micros

          std::chrono::high_resolution_clock::duration duration = std::chrono::high_resolution_clock::now().time_since_epoch();
          seed = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        }
        else
        {
          try
          {
            seed = std::stol(eaConfig[i]);
          }
          catch(std::invalid_argument e)
          {
            cout << "Error: \"" << eaConfig[i] << "\" is not a valid seed. Please only use seeds from logfiles\n"
                    " or a value of \"true\" if you want a time-generated seed" << endl;
            exit(1);
          }
        }
        cout << "Running with seed " << seed << endl;
        std::srand(seed);
        break;

    }
  }

  //Start the log file
  time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

  logFileOut << "Result Log"
             << "\n=== Config ==="
             << "\nStarted at: " << std::ctime(&time)
             << "CNF Directory: " << inputDirectory
             << "\nSolution File: " << solutionFile
             << "\nSeed: " << seed
             << "\nNumber of runs per experiment: " << numRuns
             << "\nNumber of evals per run: " << evalsPerRun
             << "\n=== End Config ==="
             << endl;

  Genome genes;
  for(int i = 0; i < numRuns; i++)
  {
    logFileOut << "Run " << (i + 1) << endl;
    cout << "================== Run " << (i+1) << " =====================" << endl;
    for(int j = 0; j < evalsPerRun; j++)
    {
      //Generate random genome
      randomGenome(genes);

      cout << "Eval #" << j + 1 << ": ";
      //Evaluate fitness
      currentFitness = evaluateFitness(inputDirectory, genes);

      cout << currentFitness << " (" << 100 - currentFitness << " sec)" << endl;
      if(currentFitness > maxFitness)
      {
        maxFitness = currentFitness;
        bestParamString = genes.getParamString();
        logFileOut << j + 1 << "\t" << currentFitness << endl;
      }
    }
    //Reset fitness to minimum possible value
    logFileOut << endl;
    maxFitness = -FLT_MAX;
  }

  //Write optimal parameters to solution file
  solutionFileOut << "./solvers/minisat/minisat " << bestParamString << " " << inputDirectory << endl;

  time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  logFileOut << "Ended at " << std::ctime(&time);
  logFileOut.close();
  solutionFileOut.close();

  return 0;
}

float evaluateFitness(const string inputDirectory, const Genome& genes)
{
    float elapsedTime = -FLT_MAX;
    int exitCode;
    string s;
    string command1 = "solvers/minisat/minisat " + genes.getParamString() + " " + inputDirectory + " > output.txt";
    string command2 = "cat output.txt | tail -3 | head -1 | cut -f17 -d ' ' > file.txt";
    exitCode = system(command1.c_str());
    system(command2.c_str());

    if(exitCode != 0)
    {
        cout << "Error: Nonzero exit code!";
        system("echo $'=====\n' >> error.log");
        system("cat output.txt >> error.log");
        return -FLT_MAX;
    }
    std::ifstream inputFile;
    inputFile.open("file.txt");

    if(inputFile)
    {
        inputFile >> s;

        try
        {
            elapsedTime = stof(s);
        }
        catch (std::invalid_argument e)
        {
            cout << "Oops... " << s << " is not what we were looking for" << endl;
            system("echo $'=====\n' >> error.log");
            system("cat output.txt >> error.log");
            return -FLT_MAX;
        }
        inputFile.close();
    }

    //cout << "Elapsed Time: " << elapsedTime << " sec" << endl;
    //cout << "Fitness: " << 100 - elapsedTime << endl;
    return 100 - elapsedTime;
}

void randomGenome(Genome& genes)
{
    //rand() / RAND_MAX -> gives a double in the range [0, 1]
    //luby or no-luby
    genes.luby = std::round(static_cast<double>(std::rand()) / RAND_MAX) == 1;

    //rnd-freq -> [0, 1]
    genes.rnd_freq = static_cast<double>(std::rand()) / RAND_MAX;

    //var-decay -> (0, 1)
    genes.var_decay = static_cast<double>(std::rand()) / RAND_MAX;

    //Make the range exclusive
    if(genes.var_decay == 0)
    {
        genes.var_decay = .0001;
    }
    else if(genes.var_decay == 1)
    {
        genes.var_decay = .9999;
    }

    //cla-decay -> (0, 1)
    genes.cla_decay = static_cast<double>(std::rand()) / RAND_MAX;

    //Make the ranges exclusive
    if(genes.cla_decay == 0)
    {
        genes.cla_decay = .0001;
    }
    else if(genes.cla_decay == 1)
    {
        genes.cla_decay = .9999;
    }

    //rinc -> (1, inf)
    genes.rinc = (DBL_MAX * (static_cast<double>(std::rand()) / RAND_MAX)) + 1.0001;

    //gc-frac -> (0, inf)
    genes.gc_frac = DBL_MAX * (static_cast<double>(std::rand()) / RAND_MAX) + .0001;

    //rfirst -> [1, INT_MAX]
    genes.rfirst = INT_MAX * (static_cast<double>(std::rand()) / RAND_MAX) + 1;

    //ccmin-mode -> {0, 1, 2}
    genes.ccmin_mode = std::rand() % 3;

    //phase-saving -> {0, 1, 2}
    genes.phase_saving = std::rand() % 3;
}