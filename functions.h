#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include <math.h>
#include <vector>
#include <deque>
#include <cassert>
#include <algorithm>

// #pragma warning(disable : 4996) //stop Visual studio giving errors about some GCC compiler differences

/* For Thomas Algorithm */
#define MAXSIZE (200+2)

/* Setting up mesh size */
#define IMAX 16  //number of nodes in the X direction
#define JMAX 40 //number of nodes in the Y direction
extern double dT;

#define MAX_ITER 2500

/* setting dimensions of the domain */
#define Width 1.0
#define Height 2.5

#define dX (Width / IMAX)
#define dY (Height / JMAX)
#define dX2 ((Width*Width) /(IMAX*IMAX))
#define dY2 ((Height*Height) /(JMAX*JMAX))

/*------defining constant flow values-------*/
#define Re 150.0
#define Beta 1.0
#define U_top 1.0
#define A_pressure 0.0
//-2.1 -2.7

/*----variables for checking corectness of flux inregral in part 1----*/
#define P0 1.0
#define u0 1.0
#define v0 1.0

#define PI 3.141592653589793238462643383279502884L /* pi */

// //macro for finding MAX of a pair
// #define MAX(a,b) ((a)> (b) ? (a) : (b))

//Tolerance for convergence
#define Tol 1.e-15

//structs for error handling
struct L2NORM {
	double _1{};
	double _2{};
	double _3{};
};

struct ERROR {
	double _1{};
	double _2{};
	double _3{};
};

/* defining external global variables for all cpp files */
extern double*** Soln, *** dSoln, ***Soln2;
extern double*** FI;

extern double**** Bx, **** Cx, **** Ax;
extern double**** By, **** Cy, **** Ay;

/* Thomas Algorithm */
void SolveBlockTri(double LHS[MAXSIZE][3][3][3], double RHS[MAXSIZE][3], int iNRows);


/* memory allocation functions */
void mem_alloc_residual(double**** Fx, double**** Fy, double**** FIx, double**** FIy);
void mem_dealloc_residual(double**** Fx, double**** Fy, double**** FIx, double**** FIy);
void mem_alloc_fluxJacobian(double***** FFJ1, double***** FFJ2, double***** GFJ1, double***** GFJ2);
void mem_dealloc_fluxJacobian(double***** FFJ1, double***** FFJ2, double***** GFJ1, double***** GFJ2);
//void mem_alloc_AF(double***** LHS, double*** RHS);
//void mem_dealloc_AF(double***** LHS, double*** RHS);
void memory_allocation();
void memory_deallocate();



/* debugging functions */
void write_col(FILE* file, const char* filename, double** array2d, int imax, int jmax);
void write_row(FILE* file, const char* filename, double*** array2d, int imax, int jmax, int component);
void write_row2(FILE* file, const char* filename, double** array2d, double** array2d2, int imax, int jmax);
void write_row_error(FILE* file, const char* filename, double** array2d, double** array2d2, double** exact, int imax, int jmax);
void write_row_1d(FILE* file, const char* filename, double array1d[IMAX + 2][3], int l, int component);

void get_l2norm(FILE* solution, int iter);

void write_tecplot(FILE* file, const char* filename, double*** array);
void write_tecplot_diff(FILE* file, const char* filename, double*** array, double*** array2);

void write_sym_plot_u(FILE* file, const char* filename, double*** array1d, int l);

template <class T , class M>
void Linear_Regression(std::vector<T> indep_var, std::deque<T> dep_var, M &a_1, M &a_2);

#endif // !FUNCTIONS_H

