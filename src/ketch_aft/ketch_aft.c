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

    FILE *fp;

    // Check only one or two command-line arguments are given (time-temperature file and write TL dist flag)
    if (argc != 2 && argc != 3) // the program's name is the first argument
    {
        printf("usage: ketch_aft tT_file [write_tl_dist_flag]\n");
        exit(1);
    }

    // Read tT file to get number of lines
    fptr = fopen(argv[1], "r");
    while(fgets(line, LSIZ, fptr))
    {
        i++;
    }
    fclose(fptr);
    tot = i;
    // Exit if tT file has >1000 lines
    if (tot > 1000) {
        printf("Error: Temperature histories with more than 1000 points are not supported.\n");
        exit(1);
    }

    float time[tot];
    float temp[tot];

    fptr = fopen(argv[1], "r"); 

    // Read tT file again and store times/temperatures
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

    // Define AFT age calculation params
    nstep = tot;
    alo = 16.3;
    final_age = 0.0;
    oldest_age = 0.0;
    fmean = 0.0;

    // Calculate AFT age/track-length distro
    ketch_main(&nstep, time, temp, &alo, &final_age, &oldest_age, &fmean, ftdist);

    printf("The final age is %f Ma (Mean track length: %f um)\n", final_age, fmean);

    // Write track-length distro to file
    int write_tl_dist_flag = 0;
    if (argc == 3)
    {
        write_tl_dist_flag = atoi(argv[2]);
    }
    if (write_tl_dist_flag == 1)
    {
        fp = fopen("ft_length.csv", "w");
        fprintf(fp, "Track length,Probability\n");
        for(i = 0; i < 200; i++)
        {
            fprintf(fp, "%f,%f\n", (i*1.0+0.5)*20.0/200,ftdist[i]);
        }
        fclose(fp);
    }

    return 0;
}
