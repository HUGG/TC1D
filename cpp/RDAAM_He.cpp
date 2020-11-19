//---------------------------------------------------------------------------

#pragma hdrstop

#include "RDAAM.h"
#include <iostream>
#include <string>
#include <fstream>

//---------------------------------------------------------------------------

#pragma argsused
// int _tmain(int argc, _TCHAR* argv[])
int main(int argc, char *argv[])
{
	double ap_age, ap_corrAge, zr_age, zr_corrAge, total_He;
	double ap_rad, ap_U, ap_Th, zr_rad, zr_U, zr_Th;
	char dummy[255];
	char c;
	std::string time, temp;

	if (argc != 2 && argc != 8) // the program's name is the first argument
    {
      std::cerr << "usage: RDAAM tTfile [ap_rad ap_U ap_Th zr_rad zr_U zr_Th]\n";
      return -1;
    }

    if (argc == 8)
	{
	  ap_rad = atof(argv[2]);
	  ap_U   = atof(argv[3]);
	  ap_Th  = atof(argv[4]);
	  zr_rad = atof(argv[5]);
	  zr_U   = atof(argv[6]);
	  zr_Th  = atof(argv[7]);
	} else
	{
	  ap_rad = 60.0;
	  ap_U   = 10.0;
	  ap_Th  = 40.0;
	  zr_rad = 60.0;
	  zr_U   = 10.0;
	  zr_Th  = 40.0;
	}

	TTPath path;
	path.clear();

	// Read in time-temperature history from file
	
	std::ifstream infile (argv[1]);
    while (std::getline(infile, time, ','))
	{
	    std::getline(infile, temp, '\n') ;
    	TTPathPoint pt = {std::stof(time),std::stof(temp)};
    	path.push_back(pt);
	}

	//for(int i = 1; i < (argc); i+=2)
    //{
    //   // arguments are like 'hello[500]', 'array[100]
    //    TTPathPoint pt = {atof(argv[i]),atof(argv[i+1])};
    //    path.push_back(pt);
    //}

	//TTPathPoint end = {0.0, 0.0};   // time (Ma), temperature (C)
	//TTPathPoint mid = {10.0, 40.0};
	//TTPathPoint start = {100.0, atof(argv[1])};

	//path.push_back(end);   // Present day is first entry in path
	//path.push_back(mid);
	//path.push_back(start); // Earliest t-T point is last entry in path

// Run this first to set up the model.  Run again if you change one of the parameters
	RDAAM_Init(HE_PREC_BEST, ap_rad, ap_U, ap_Th, 0.0); // precision, radius, U, Th, Sm

// After RDAAM_Init is run, run this as many times as you want for each path to model
// Set final parameter (optimize) to true for geological histories, false for degassing
	int ap_success = RDAAM_Calculate(&path, ap_age, ap_corrAge, total_He, false);

// For zircon, just change the first letter
	ZRDAAM_Init(HE_PREC_BEST, zr_rad, zr_U, zr_Th, 0.0); // precision, radius, U, Th, Sm
	int zr_success = RDAAM_Calculate(&path, zr_age, zr_corrAge, total_He, false);

// Once you are done with this calculation, run this to clean up.
	RDAAM_FreeCalcArrays();

	if (ap_success == 1) printf("Apatite Age = %.2lf, Corrected age = %.2lf\n",ap_age,ap_corrAge);
	else printf("Something went wrong...\n");

	if (zr_success == 1) printf("Zircon Age = %.2lf, Corrected age = %.2lf\n",zr_age,zr_corrAge);
	else printf("Something went wrong...\n");

	//printf("Hit <enter> to continue\n");
	//while ((c = getchar()) != '\n') ;

	return 0;
}
//---------------------------------------------------------------------------
