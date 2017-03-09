#ifndef GENOME_H
#define GENOME_H
#include <iostream>
#include <limits>
#include <iomanip>

class Genome
{
  public:
    bool luby;
    double rnd_freq;
    double var_decay;
    double cla_decay;
    double rinc;
    double gc_frac;
    int rfirst;
    int ccmin_mode;
    int phase_saving;
    double wallTimeTrainingSubfitness;
    double wallTimeTestSubfitness;
    int memoryUsageTrainingSubfitness;
    int memoryUsageTestSubfitness;
    double numDecisionsTrainingSubfitness;
    double numDecisionsTestSubfitness;
    bool enteredInTournament;
    bool chosen;
    int id;
    double selectionChance;
    double sigma[9];
    int paretoLevel;

    Genome() : luby(false), rnd_freq(0), var_decay(0.0001), cla_decay(0.0001), rinc(1.0001), gc_frac(0.0001), rfirst(1),
               ccmin_mode(0), phase_saving(0), wallTimeTrainingSubfitness(std::numeric_limits<double>::lowest()),
               wallTimeTestSubfitness(std::numeric_limits<double>::lowest()),
               memoryUsageTrainingSubfitness(std::numeric_limits<int>::lowest()),
               memoryUsageTestSubfitness(std::numeric_limits<int>::lowest()),
               numDecisionsTrainingSubfitness(std::numeric_limits<double>::lowest()),
               numDecisionsTestSubfitness(std::numeric_limits<double>::lowest()),
               enteredInTournament(false), chosen(false),
               id(-1), selectionChance(0), paretoLevel(0)
    {
      for(int i = 0; i < 9; i++)
      {
        sigma[i] = 0;
      }
    }

    bool dominates(const Genome& other, bool testSet) const
    {
      if(this != &other)
      {
        if(testSet)  
        {
          if(wallTimeTestSubfitness >= other.wallTimeTestSubfitness && 
              memoryUsageTestSubfitness >= other.memoryUsageTestSubfitness &&
              numDecisionsTestSubfitness >= other.numDecisionsTestSubfitness)
          {
            return !(wallTimeTestSubfitness == other.wallTimeTestSubfitness && 
              memoryUsageTestSubfitness == other.memoryUsageTestSubfitness &&
              numDecisionsTestSubfitness == other.numDecisionsTestSubfitness);
          }
        }
        else
        {
          if(wallTimeTrainingSubfitness >= other.wallTimeTrainingSubfitness &&
             memoryUsageTrainingSubfitness >= other.memoryUsageTrainingSubfitness &&
             numDecisionsTrainingSubfitness >= other.numDecisionsTrainingSubfitness)
          {
            return !(wallTimeTrainingSubfitness == other.wallTimeTrainingSubfitness &&
                     memoryUsageTrainingSubfitness == other.memoryUsageTrainingSubfitness &&
                     numDecisionsTrainingSubfitness == other.numDecisionsTrainingSubfitness);
          }
        }
      }
      
      return false;  
    }
    
    std::string getParamString() const
    {
      std::ostringstream out;
      out << "-" << (luby ? "" : "no-")
          << "luby"
          << " -rnd-freq=" << std::setprecision(15) << rnd_freq
          << " -var-decay=" << std::setprecision(15) << var_decay
          << " -cla-decay=" << std::setprecision(15) << cla_decay
          << " -rinc=" << std::setprecision(15) << rinc
          << " -gc-frac=" << std::setprecision(15) << gc_frac
          << " -rfirst=" << rfirst
          << " -ccmin-mode=" << ccmin_mode
          << " -phase-saving=" << phase_saving;
      return out.str();
    };

    friend bool operator<(const Genome& genome1, const Genome& genome2)
    {
      return genome1.id < genome2.id;
    }

    friend bool operator==(const Genome& genome1, const Genome& genome2)
    {
      return genome1.id == genome2.id
             && genome1.luby == genome2.luby
             && genome1.ccmin_mode == genome2.ccmin_mode
             && genome1.cla_decay == genome2.cla_decay
             && genome1.gc_frac == genome2.gc_frac
             && genome1.phase_saving == genome2.phase_saving
             && genome1.rfirst
             && genome1.rinc == genome2.rinc
             && genome1.rnd_freq == genome2.rnd_freq
             && genome1.var_decay == genome2.var_decay;
    }

    friend bool operator>(const Genome& genome1, const Genome& genome2)
    {
      return genome1.id > genome2.id;
    }
};
#endif