//---------------------------------------------------------------------------

#ifndef RDAAMH
#define RDAAMH
//---------------------------------------------------------------------------
#include <math.h>
#include <vector>

#define HE_PREC_GOOD     0
#define HE_PREC_BETTER   1
#define HE_PREC_BEST     2

/* Useful constants */
#define PI 3.1415926535
/* Seconds per year from 11th General Conference on Weights and Measures, 1960 */
#define	SECS_PER_MA		3.15569259747e13
#define	SECS_PER_YR		3.15569259747e7
#define	KELVINS_AT_0C	273.15
#define SQRT2PI	2.50662827463
// Universal gas constant cal/K/mol
#define UNIV_GAS_CONST 1.987
//#define UNIV_GAS_CONST 1.98588

/* Decay constants */
#define	U238YR	1.55125e-10
#define	U238MA	1.55125e-4
#define U238SEC 4.91575e-18
#define U235YR  9.8485e-10
#define U235MA  9.8485e-4
#define U235SEC 3.12089e-17
#define TH232YR 4.9475e-11
#define TH232MA 4.9475e-5
#define TH232SEC 1.56781e-18
#define SM147YR 6.54e-12
#define SM147MA 6.54e-6
#define SM147SEC 2.072e-19

/* Fission decay constant */
#define U238YR_SF 8.46e-17

/* ppm-nmol/g conversions */
#define PPM_NMPG_U238  4.20168
#define PPM_NMPG_U235  4.25532
#define PPM_NMPG_TH232 4.31034
#define PPM_NMPG_SM147 6.80272

// U238/U235 ratio
#define U238U235RATIO   137.88

/* moles */
#define N_PER_MOL   6.02214199e23
#define N_PER_NMOL	6.02214199e14

/* Densities in g/cm3 */
#define DENSITY_APATITE	3.19

// Units for psi (trapping model normalization factor)
#define PSI_G_NMOL			0
#define PSI_CMSQ_TRACK	1
// For Guenthner et al, in which this is a diffusivity factor
#define PSI_D0N17_1_SEC	2
/* Direct-impact law amorphous material per decay in zircon in g/alpha */
#define B_ALPHA 5.48e-19

using namespace std;

typedef struct TTPathPointType {
  double time,temperature;
} TTPathPoint;
typedef vector<TTPathPoint>  TTPath;

typedef struct annealParamType {
	double c0,c1,c2,c3,a,b,lMin;
} annealParamRec;

typedef struct alphaStoppingDistType {
	double asU238;
	double asU235;
	double asTh232;
	double asSm147;
} alphaStoppingDistRec;


void GeneralInit(int, double, double, double, double);
void RDAAM_Init(int, double, double, double, double);
void ZRDAAM_Init(int, double, double, double, double);
void SetPrecision(int);
bool RDAAM_CheckParameters();
void RDAAM_PrepModel();
void RDAAM_CalcAlphaCorrectionFactor();
int RDAAM_InterpolateTTPath(TTPath *,double);
void RDAAM_ExtractHeProfile();
void RDAAM_InitFTAnnealingTraps(double ** &, bool);
void RDAAM_CalcHeAge(bool);
void RDAAM_FreeCalcArrays();
void nrerror(char *);
void dtridag(double [], double [], double [], double [], double [], double [], int);
void dtridag2(double [], double [], double [], double [], int);
void dpolint2(double [], double [], int, double, double *, double *);
double *dvector(int, int);
void free_dvector(double *, int);
double **dmatrix(int, int, int, int);
void free_dmatrix(double **, int, int, int, int);
int	RDAAM_Calculate(TTPath *, double &, double &, double &, bool);

#endif
