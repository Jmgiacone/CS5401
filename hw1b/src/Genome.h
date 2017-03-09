//
// Created by jordan on 9/2/16.
//

#ifndef HW1A_GENOME_H
#define HW1A_GENOME_H
#include <iostream>

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
    float trainingFitness;
    float testFitness;
    bool enteredInTournament;
    int id;
    float selectionChance;


    Genome() : luby(false), rnd_freq(0), var_decay(0.0001), cla_decay(0.0001), rinc(1.0001), gc_frac(0.0001), rfirst(1),
               ccmin_mode(0), phase_saving(0), trainingFitness(-FLT_MAX), testFitness(-FLT_MAX),
               enteredInTournament(false), id(-1), selectionChance(0) {}

    std::string getParamString() const
    {
        std::string s = "-";

        s += (luby ? "" : "no-");
        s += "luby";
        s += " -rnd-freq=" + std::to_string(rnd_freq);
        s += " -var-decay=" + std::to_string(var_decay);
        s += " -cla-decay=" + std::to_string(cla_decay);
        s += " -rinc=" + std::to_string(rinc);
        s += " -gc-frac=" + std::to_string(gc_frac);
        s += " -rfirst=" + std::to_string(rfirst);
        s += " -ccmin-mode=" + std::to_string(ccmin_mode);
        s += " -phase-saving=" + std::to_string(phase_saving);

        return s;
    };
};


#endif