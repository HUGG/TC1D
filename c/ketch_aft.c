#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ketch.h"

#define LSIZ 128 

int main( int argc, char *argv[] )  {

    int nstep;
    double alo, final_age, oldest_age, fmean;
    double ftdist[200] = { 0.0 };;

    char line[LSIZ];
    FILE *fptr = NULL; 
    int i = 0;
    int tot = 0;
    char *pt;

	if (argc != 2) // the program's name is the first argument
    {
        printf("usage: ketch_aft tT_file\n");
        exit(1);
    }

    fptr = fopen(argv[1], "r");
    while(fgets(line, LSIZ, fptr))
    {
        i++;
    }
    fclose(fptr);
    tot = i;
    if (tot > 1000) {
        printf("Error: Temperature histories with more than 1000 points are not supported.\n");
        exit(1);
    }

    float time[tot];
    float temp[tot];

    fptr = fopen(argv[1], "r"); 

    for(i = 0; i < tot; ++i)
    {
        fgets(line, LSIZ, fptr);
        pt = strtok(line,",");
        time[i] = atof(pt);
        while (pt != NULL) {
            temp[i] = atof(pt);
            pt = strtok (NULL, ",");
        }
        /* printf("Time is %f and temp is %f\n", time[i], temp[i]); */
    }
    fclose(fptr);

    nstep = tot;
    alo = 16.0;
    final_age = 0.0;
    oldest_age = 0.0;
    fmean = 0.0;

    ketch_main(&nstep, time, temp, &alo, &final_age, &oldest_age, &fmean, ftdist);

    printf("The final age is %f Ma\n", final_age);
    return 0;
}
