//
// Created by jordan on 9/2/16.
//

#ifndef HW1A_GENOME_H
#define HW1A_GENOME_H
#include <iostream>
const char* const params[] {"-rnd-freq", "-var-decay", "-cla-decay", "-rinc", "-gc-frac", "-rfirst", "-ccmin-mode", "-phase-saving"};
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


    Genome() : luby(false), rnd_freq(0), var_decay(0.1), cla_decay(0.1), rinc(1.1), gc_frac(0.1), rfirst(1), ccmin_mode(0), phase_saving(0) {}
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