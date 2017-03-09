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
    double trainingFitness;
    double testFitness;
    bool enteredInTournament;
    bool chosen;
    int id;
    double selectionChance;
    double sigma[9];

    Genome() : luby(false), rnd_freq(0), var_decay(0.0001), cla_decay(0.0001), rinc(1.0001), gc_frac(0.0001), rfirst(1),
               ccmin_mode(0), phase_saving(0), trainingFitness(std::numeric_limits<double>::lowest()),
               testFitness(std::numeric_limits<double>::lowest()), enteredInTournament(false), chosen(false), id(-1),
               selectionChance(0)
    {
      for(int i = 0; i < 9; i++)
      {
        sigma[i] = 0;
      }
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
};
#endif