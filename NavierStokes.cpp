#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "functions.h"
#include "DMD.h"

#include <petscksp.h>
#include <slepcsvd.h>
#include <slepceps.h>

struct _snapshots_type{
	Mat mat=PETSC_NULL;
	PetscInt iNumRows, iNumCols;
	std::vector<PetscInt> col_index, row_indices;
	std::vector<PetscInt> old_row_indices, new_row_indices;
	std::vector<PetscInt> last_row_index; // REMOVE THIS
	std::vector<PetscScalar> vValues; // Vec values
	std::vector<PetscScalar> mValues; // Mat values
	};


/* Time Advance option */
PetscErrorCode TimeAdvance(PetscInt &isnapFreq, PetscInt &nDMD, PetscInt& numIters, PetscInt*& dmdIter, PetscBool& flg_DMD);
PetscErrorCode TimeAdvanceSmart(PetscInt &isnapFreq, PetscInt &numIters, PetscInt *&dmdIter,
		PetscInt &iDummyDMDIter, PetscBool &flg_DMD);

/* DMD added Functions */
PetscErrorCode readOpts(PetscReal &dT, PetscInt &isnapFreq, PetscInt &max_iter_total,
		PetscBool &flg_DMD, PetscBool &flg_dmdAuto, PetscInt &nDMD,
		PetscInt *&dmdIter, PetscInt& iDummyDMDIter);
PetscErrorCode initSnapsMat(Vec& vec, _snapshots_type& snap);
PetscErrorCode updateSolutionMatrix(Vec& vec, _snapshots_type& snap);

/* -----  Print functions  ------ */
PetscErrorCode printMatMATLAB(std::string sFilename,
		std::string sMatrixName, Mat A);
PetscErrorCode printVecMATLAB(std::string sFileName,
		std::string sVectorName, Vec V);
PetscErrorCode printVecPYTHON(std::string sFileName,
		std::string sVectorName, Vec V);
PetscErrorCode printMatPYTHON(std::string sFilename,
		std::string sMatrixName, Mat A);

/* declaring global variables */
double*** Soln{}, *** dSoln{}, ***Soln2{};
double*** FI{};

/* 4D arrays */
double**** Bx{}, **** Cx{}, **** Ax{};
double**** By{}, **** Cy{}, **** Ay{};

double dT{0.05};


/* -----------Initializing the solution-------- */
void initialize(double*** Soln, double*** dSoln) {
	for (int i = 0; i < IMAX + 2; i++) {
		double X = (i - 0.5) * dX;
		for (int j = 0; j < JMAX + 2; j++) {
			double Y = (j - 0.5) * dY;
			Soln[i][j][0] = P0 * cos(PI * X) * cos(PI * Y);
			Soln[i][j][1] = u0 * sin(PI * X) * sin(2 * PI * Y);
			Soln[i][j][2] = my_v0 * sin(2 * PI * X) * sin(PI * Y);
		}
	}
	/* for validation part */
	///* Initializing changes of parameters */
	//for (int i = 0; i < IMAX + 2; i++) {
	//	for (int j = 0; j < JMAX + 2; j++) {
	//		dSoln[i][j][0] = 0;
	//		dSoln[i][j][1] = 0;
	//		dSoln[i][j][2] = 0;
	//	}
	//}
	//{
	//	int i{ 10 }, j{ 10 }; 
	//	dSoln[i][j][0] = 1e-6; //1st column
	//	dSoln[i][j][1] = 1e-6; //2nd column
	//	dSoln[i][j][2] = 1e-6; //3rd column Jacobian
	//}
}

/* ---------applying BCs by setting ghost cell values---------- */
void ApplyBC(double*** Soln) {
	/* no-slip BC at bottom of the domain */
	for (int i = 0; i < IMAX + 2; i++) {
		Soln[i][0][0] =  Soln[i][1][0]; // P
		Soln[i][0][1] = -Soln[i][1][1]; // u
		Soln[i][0][2] = -Soln[i][1][2]; // v
	}
	/* no-slip BC at top of the domain, with constant velocity U_top */
	for (int i = 0; i < IMAX + 2; i++) {
		Soln[i][JMAX + 1][0] =  Soln[i][JMAX][0];			 // P
		Soln[i][JMAX + 1][1] = 2.0 * U_top - Soln[i][JMAX][1]; // u
		Soln[i][JMAX + 1][2] = -Soln[i][JMAX][2];			 // v
	}

	/* no-slip BC at left and right of the domain  */
	for (int j = 0; j < JMAX + 2; j++) {
		/* ---------left wall--------- */
		Soln[0][j][0] =  Soln[1][j][0]; // P
		Soln[0][j][1] = -Soln[1][j][1]; // u
		Soln[0][j][2] = -Soln[1][j][2]; // v
		/* ---------right wall-------- */
		Soln[IMAX + 1][j][0] =  Soln[IMAX][j][0]; // P
		Soln[IMAX + 1][j][1] = -Soln[IMAX][j][1]; // u
		Soln[IMAX + 1][j][2] = -Soln[IMAX][j][2]; // v
	}
}


/* ----------------calculating the RHS (residual) of our system of equations------------ */
void calc_residual(double*** Soln, double*** FI) {
	/*double Fx[IMAX + 2][JMAX + 2][3]{};
	double Fy[IMAX + 2][JMAX + 2][3]{};
	double FIx[IMAX][JMAX][3]{};
	double FIy[IMAX][JMAX][3]{};*/
	double*** Fx{}, *** Fy{}, *** FIx{}, *** FIy{};

	mem_alloc_residual(&Fx, &Fy, &FIx, &FIy);

	/* calculating flux in the interior of the domain - X direction */
	for (int j = 1; j < JMAX + 1; j++) {//don't calculate flux in the ghost cell region, hence the loops are -2
		for (int i = 1; i < IMAX + 2; i++) {
			int I_ = i - 1;
			Fx[i][j][0] = (Soln[I_][j][1] + Soln[I_ + 1][j][1]) / (2.0 * Beta);

			Fx[i][j][1] = ((Soln[I_][j][1] + Soln[I_ + 1][j][1]) / 2.0) * ((Soln[I_][j][1] + Soln[I_ + 1][j][1]) / 2.0)
				+ (Soln[I_][j][0] + Soln[I_ + 1][j][0]) / 2.0
				- (Soln[I_ + 1][j][1] - Soln[I_][j][1]) / (Re * dX);

			Fx[i][j][2] = ((Soln[I_][j][1] + Soln[I_ + 1][j][1]) / 2.0) * ((Soln[I_][j][2] + Soln[I_ + 1][j][2]) / (2.0))
				- (Soln[I_ + 1][j][2] - Soln[I_][j][2]) / (Re * dX);
		}

	}
	/* calculating flux integral using the evaluated fluxes - X direction */
	for (int j = 0; j < JMAX; j++) {
		for (int i = 0; i < IMAX; i++) {
			int I_ = i + 2;
			int J = j + 1;
			FIx[i][j][0] = (Fx[I_][J][0] - Fx[I_ - 1][J][0]) / dX;
			FIx[i][j][1] = (Fx[I_][J][1] - Fx[I_ - 1][J][1]) / dX;
			FIx[i][j][2] = (Fx[I_][J][2] - Fx[I_ - 1][J][2]) / dX;
		}
	}

	/* calculating flux in the interior - Y direction */
	for (int i = 1; i < IMAX + 1; i++) {
		for (int j = 1; j < JMAX + 2; j++) {
			int J = j - 1;
			Fy[i][j][0] = (Soln[i][J][2] + Soln[i][J + 1][2]) / (2.0 * Beta);

			Fy[i][j][1] = ((Soln[i][J][1] + Soln[i][J + 1][1]) / 2.0) * ((Soln[i][J][2] + Soln[i][J + 1][2]) / 2.0)
				- (Soln[i][J + 1][1] - Soln[i][J][1]) / (Re * dY);

			Fy[i][j][2] = ((Soln[i][J][2] + Soln[i][J + 1][2]) / 2.0) * ((Soln[i][J][2] + Soln[i][J + 1][2]) / 2.0)
				+ (Soln[i][J][0] + Soln[i][J + 1][0]) / 2.0
				- (Soln[i][J + 1][2] - Soln[i][J][2]) / (Re * dY);
		}
	}

	/* calculating flux integrals using the evaluated fluxes - Y direction */
	for (int i = 0; i < IMAX; i++) {
		for (int j = 0; j < JMAX; j++) {
			int J = j + 2;
			int I_ = i + 1;
			FIy[i][j][0] = (Fy[I_][J][0] - Fy[I_][J - 1][0]) / dY;
			FIy[i][j][1] = (Fy[I_][J][1] - Fy[I_][J - 1][1]) / dY;
			FIy[i][j][2] = (Fy[I_][J][2] - Fy[I_][J - 1][2]) / dY;
		}
	}

	/* calculating the residual using the evaluated flux integrals */
	/* these include dT and negative sign of going to the RHS of our equation */
	for (int i = 1; i < IMAX+1; i++) {
		for (int j = 1; j < JMAX+1; j++) {
			// defining pressure oscillation fix
			double pressure_fix = A_pressure * dX * dY * ((Soln[i - 1][j][0] - 2.0 * Soln[i][j][0] + Soln[i + 1][j][0]) / (dX * dX) +
				(Soln[i][j - 1][0] - 2 * Soln[i][j][0] + Soln[i][j + 1][0]) / (dY * dY));

			FI[i][j][0] = -dT * (FIx[i-1][j-1][0] + FIy[i-1][j-1][0] + pressure_fix);
			FI[i][j][1] = -dT * (FIx[i-1][j-1][1] + FIy[i-1][j-1][1]);
			FI[i][j][2] = -dT * (FIx[i-1][j-1][2] + FIy[i-1][j-1][2]);
		}
	}
	mem_dealloc_residual(&Fx, &Fy, &FIx, &FIy);
}

/* -----------calculating      Ax, Bx, Cx ------and----- Ay, By, Cy   ---------- */
void calc_fluxJacobians(double*** Soln, double**** Bx, double**** Cx, double**** Ax,
	double**** By, double**** Cy, double**** Ay) {
	/*double FFJ1[IMAX + 2][JMAX + 2][3][3]{}, FFJ2[IMAX + 2][JMAX + 2][3][3]{};
	double GFJ1[IMAX + 2][JMAX + 2][3][3]{}, GFJ2[IMAX + 2][JMAX + 2][3][3]{};*/

	double**** FFJ1{}, ****FFJ2{};
	double**** GFJ1{}, ****GFJ2{};

	mem_alloc_fluxJacobian(&FFJ1, &FFJ2, &GFJ1, &GFJ2);

	/* Calculating 2 types of Flux Jacobian for X direction */
	for (int j = 1; j < JMAX + 1; j++) {
		for (int i = 1; i < IMAX + 2; i++) {
			int I_ = i - 1;
			/* Defining Jacobian type 1: F(i+0.5,j)/U(i,j) and F(i-0.5,j)/U(i-1,j) */
			FFJ1[i][j][0][0] = 0;	FFJ1[i][j][0][1] = 1.0 / (2.0 * Beta);											FFJ1[i][j][0][2] = 0;
			FFJ1[i][j][1][0] = 0.5;	FFJ1[i][j][1][1] = (Soln[I_][j][1] + Soln[I_ + 1][j][1]) / 2.0 + 1.0 / (dX * Re); FFJ1[i][j][1][2] = 0;
			FFJ1[i][j][2][0] = 0;	FFJ1[i][j][2][1] = (Soln[I_][j][2] + Soln[I_ + 1][j][2]) / 4.0;				    FFJ1[i][j][2][2] = (Soln[I_][j][1] + Soln[I_ + 1][j][1]) / 4.0 + 1.0 / (dX * Re);
			/* Defining Jacobian type 2: F(i+0.5,j)/U(i+1,j) and F(i-0.5,j)/U(i,j) */
			FFJ2[i][j][0][0] = 0;	FFJ2[i][j][0][1] = 1.0 / (2.0 * Beta);										    FFJ2[i][j][0][2] = 0;
			FFJ2[i][j][1][0] = 0.5;	FFJ2[i][j][1][1] = (Soln[I_][j][1] + Soln[I_ + 1][j][1]) / 2.0 - 1.0 / (dX * Re); FFJ2[i][j][1][2] = 0;
			FFJ2[i][j][2][0] = 0;	FFJ2[i][j][2][1] = (Soln[I_][j][2] + Soln[I_ + 1][j][2]) / 4.0;				    FFJ2[i][j][2][2] = (Soln[I_][j][1] + Soln[I_ + 1][j][1]) / 4.0 - 1.0 / (dX * Re);
		}
	}
	/* Calculating 2 types of Flux Jacobian for Y direction */
	for (int i = 1; i < IMAX + 1; i++) {
		for (int j = 1; j < JMAX + 2; j++) {
			int J = j - 1;
			/* Defining Jacobian type 1: F(i,j+0.5)/U(i,j) and F(i,j-0.5)/U(i,j-1) */
			GFJ1[i][j][0][0] = 0;		GFJ1[i][j][0][1] = 0;															GFJ1[i][j][0][2] = 1.0 / (2.0 * Beta);
			GFJ1[i][j][1][0] = 0;		GFJ1[i][j][1][1] = (Soln[i][J][2] + Soln[i][J + 1][2]) / 4.0 + 1.0 / (dY * Re); GFJ1[i][j][1][2] = (Soln[i][J][1] + Soln[i][J + 1][1]) / 4.0;
			GFJ1[i][j][2][0] = 0.5;		GFJ1[i][j][2][1] = 0;															GFJ1[i][j][2][2] = (Soln[i][J][2] + Soln[i][J + 1][2]) / 2.0 + 1.0 / (dY * Re);
			/* Defining Jacobian type 2: F(i,j+0.5)/U(i,j+1) and F(i,j-0.5)/U(i,j) */
			GFJ2[i][j][0][0] = 0;		GFJ2[i][j][0][1] = 0;															GFJ2[i][j][0][2] = 1.0 / (2.0 * Beta);
			GFJ2[i][j][1][0] = 0;		GFJ2[i][j][1][1] = (Soln[i][J][2] + Soln[i][J + 1][2]) / 4.0 - 1.0 / (dY * Re); GFJ2[i][j][1][2] = (Soln[i][J][1] + Soln[i][J + 1][1]) / 4.0;
			GFJ2[i][j][2][0] = 0.5;		GFJ2[i][j][2][1] = 0;															GFJ2[i][j][2][2] = (Soln[i][J][2] + Soln[i][J + 1][2]) / 2.0 - 1.0 / (dY * Re);
		}
	}

	/* Iterating over mesh nodes for calculating Ax, Bx, and Cx */
	for (int i = 1; i < IMAX + 1; i++) {     //size of Ax Bx Cx : (IMAX + 2)*(JMAX + 2)
		for (int j = 1; j < JMAX + 1; j++) {

			/* iterating over Flux Jacobian indices */
			for (int k = 0; k < 3; k++) {
				for (int z = 0; z < 3; z++) {

					Bx[i][j][k][z] = (FFJ1[i + 1][j][k][z] - FFJ2[i][j][k][z]) / dX;
					Cx[i][j][k][z] = (FFJ2[i + 1][j][k][z]) / dX;
					Ax[i][j][k][z] = -(FFJ1[i][j][k][z]) / dX;

					By[i][j][k][z] = (GFJ1[i][j + 1][k][z] - GFJ2[i][j][k][z]) / dY;
					Cy[i][j][k][z] = (GFJ2[i][j + 1][k][z]) / dY;
					Ay[i][j][k][z] = -(GFJ1[i][j][k][z]) / dY;
				}
			}
		}
	}
	mem_dealloc_fluxJacobian(&FFJ1, &FFJ2, &GFJ1, &GFJ2);
}

/* -------Calculating LHS for approximate factorization stage 1 : iterating over rows------ */
void setupLHS_forRows(double LHS[IMAX + 2][3][3][3], const int Ncols, double*** Soln, const int j,
	double**** Bx, double**** Cx, double**** Ax) {

	for (int i = 1; i < Ncols + 1; i++) {
		for (int k = 0; k < 3; k++) {
			for (int z = 0; z < 3; z++) {
				/* Ax */
				LHS[i][0][k][z] = dT * Ax[i][j][k][z];	

				/* Bx */
					if (k == z) 
						LHS[i][1][k][z] = 1.0 + dT * Bx[i][j][k][z];  //adding I [identity matrix] here
					else
						LHS[i][1][k][z] = dT * Bx[i][j][k][z];
				/* Cx */
				LHS[i][2][k][z] = dT * Cx[i][j][k][z]; 
			}
		}
	}
	/* discarding variables that are outside the approximate factorization matrix */
	for (int k = 0; k < 3; k++) {
		for (int z = 0; z < 3; z++) {
			LHS[0][0][k][z] = 0.0;
			LHS[Ncols + 1][2][k][z] = 0.0;
		}
	}

	/* applying implicit BCs */
	for (int k = 0; k < 3; k++) {
		for (int z = 0; z < 3; z++) {
			if (k == z) {
				LHS[0][1][k][z] = 1.0;
				LHS[0][2][k][z] = 1.0;
				LHS[Ncols + 1][1][k][z] = 1.0;
				LHS[Ncols + 1][0][k][z] = 1.0;
			}
			else {
				LHS[0][1][k][z] = 0.0;
				LHS[0][2][k][z] = 0.0;
				LHS[Ncols + 1][1][k][z] = 0.0;
				LHS[Ncols + 1][0][k][z] = 0.0;
			}
		}
	}
	LHS[0][2][0][0] = -1.0;
	LHS[Ncols + 1][0][0][0] = -1.0;
}

/* ------Calculating LHS for approximate factorization stage 2 : iterating over columns------ */
void setupLHS_forColumns(double LHS[JMAX + 2][3][3][3], const int Nrows, double*** Soln, const int i,
	double**** By, double**** Cy, double**** Ay) {

	for (int j = 1; j < Nrows + 1; j++) {
		for (int k = 0; k < 3; k++) {
			for (int z = 0; z < 3; z++) {
				/* Ay */
				LHS[j][0][k][z] = dT * Ay[i][j][k][z];	

				/* By */
				if (k == z) 
					LHS[j][1][k][z] = 1.0 + dT * By[i][j][k][z];  //adding I [identity matrix] here
				else
					LHS[j][1][k][z] = dT * By[i][j][k][z];

				/* Cy */
				LHS[j][2][k][z] = dT * Cy[i][j][k][z]; 
			}
		}
	}
	/* discarding variables that are outside the approximate factorization matrix */
	for (int k = 0; k < 3; k++) {
		for (int z = 0; z < 3; z++) {
			LHS[0][0][k][z] = 0.0;
			LHS[Nrows + 1][2][k][z] = 0.0;
		}
	}
	/* applying implicit BCs */
	for (int k = 0; k < 3; k++) {
		for (int z = 0; z < 3; z++) {
			if (k == z) {
				LHS[0][1][k][z] = 1.0;
				LHS[0][2][k][z] = 1.0;
				LHS[Nrows + 1][1][k][z] = 1.0;
				LHS[Nrows + 1][0][k][z] = 1.0;
			}
			else {
				LHS[0][1][k][z] = 0.0;
				LHS[0][2][k][z] = 0.0;
				LHS[Nrows + 1][1][k][z] = 0.0;
				LHS[Nrows + 1][0][k][z] = 0.0;
			}
		}
	}
	LHS[0][2][0][0] = -1.0;
	LHS[Nrows + 1][0][0][0] = -1.0;
}

/* -------------Implicit Time Advance using Approximate Factorization------------ */
double ApproximateFactorization(double*** Soln, double*** FI) {

	double MaxChange = 0;
	double LHS[JMAX + 2][3][3][3]{};
	double RHS[JMAX + 2][3]{};

	/* can use dunamic arrays for this part, but didn't want to change the function arguments */
	/* in the tri-test.cpp, but the allocating and deallocating functions are available in allocation_IO.cpp */
	//double**** LHS{};
	//double** RHS{};
	//mem_alloc_AF(&LHS, &RHS); //LHS and RHS size is (JMAX+2), for the first index
	//

	/* number of columns and rows of the interior domain, for using in 1st and 2nd stage of AF */
	int Ncolumns = IMAX;
	int Nrows = JMAX;
	/*-----   STAGE 1: solving the system for interior rows   ------*/
	/* solving block by block and going on--- each loop solves one block-tri-diag matrix */
	//for every row in the mesh, which would be JMAX loops
	for (int j = 1; j < JMAX + 1; j++) {

		/* Implicit BCs are applied inside this function */
		setupLHS_forRows(LHS, Ncolumns, Soln, j, Bx, Cx, Ax);

		/* as we are iterating through rows, j is constant, and i is changing for each row */
		/* so for RHS, we have different i components at a constant j each loop for solving block tri-diag */

		/* setting RHS */
		{
			for (int i = 1; i < IMAX + 1; i++) {
				RHS[i][0] = FI[i][j][0];
				RHS[i][1] = FI[i][j][1];
				RHS[i][2] = FI[i][j][2];
			}
			/* including implicit BCs */
			RHS[0][0] = 0.;
			RHS[0][1] = 0.;
			RHS[0][2] = 0.;

			RHS[IMAX + 1][0] = 0.;
			RHS[IMAX + 1][1] = 0.;
			RHS[IMAX + 1][2] = 0.;
		}

		SolveBlockTri(LHS, RHS, Ncolumns + 2);

		/* copying results (dSoln_Tilde) to FI matrix */
		for (int i = 0; i < IMAX + 2; i++) {
			FI[i][j][0] = RHS[i][0];
			FI[i][j][1] = RHS[i][1];
			FI[i][j][2] = RHS[i][2];
		}
	}

	/*-----   STAGE 2: solving the system for interior columns   ------*/
	for (int i = 1; i < IMAX + 1; i++) {

		/* Implicit BCs are included inside the function */
		setupLHS_forColumns(LHS, Nrows, Soln, i, By, Cy, Ay);

		/* setting RHS */
		{
			for (int j = 1; j < JMAX + 1; j++) {
				RHS[j][0] = FI[i][j][0];
				RHS[j][1] = FI[i][j][1];
				RHS[j][2] = FI[i][j][2];
			}
			RHS[0][0] = 0.;
			RHS[0][1] = 0.;
			RHS[0][2] = 0.;

			RHS[JMAX + 1][0] = 0.;
			RHS[JMAX + 1][1] = 0.; //2.0 * U_top;
			RHS[JMAX + 1][2] = 0.;
		}

		SolveBlockTri(LHS, RHS, Nrows + 2);

		/* copying results (dSoln) to FI matrix */
		for (int j = 0; j < JMAX + 2; j++) {
			FI[i][j][0] = RHS[j][0];
			FI[i][j][1] = RHS[j][1];
			FI[i][j][2] = RHS[j][2];
		}
	}
	//advancing one time-step
	for (int i = 1; i <= IMAX; i++) {
		for (int j = 1; j <= JMAX; j++) {
			for (int k = 0; k < 3; k++) {
				Soln[i][j][k] += FI[i][j][k];
				MaxChange = MAX(MaxChange, fabs(FI[i][j][k]));
			}
		}
	}
	return MaxChange;
}

			/*------------------------------------------------------------------------------------------------------*/
			/*------------------------------------------- Main Program ---------------------------------------------*/
			/*------------------------------------------------------------------------------------------------------*/

int main(int argc, char **argv) {
	PetscErrorCode ierr;
	FILE* out{};

	ierr = PetscInitialize(&argc, &argv, (char*) 0, (char*) 0); CHKERRQ(ierr);
	ierr = SlepcInitialize(&argc, &argv, (char*) 0, (char*) 0); CHKERRQ(ierr);
	if (ierr)
		return ierr;

	PetscInt numIters, nDMD, *dmdIter, iDummyDMDIter, isnapFreq;
	PetscBool flg_DMD, flg_dmdAuto;

	auto start = std::chrono::steady_clock::now();

	ierr = readOpts(dT, isnapFreq, numIters, flg_DMD, flg_dmdAuto, nDMD, dmdIter, iDummyDMDIter); CHKERRQ(ierr);
	printf("\ndummy dmd iter is: %3d\n", iDummyDMDIter);

	memory_allocation();

	/* initializing P, u, and v with the initial conditions */
	initialize(Soln, dSoln);

	/* ------------ Manually configure DMD ----------------- */
	if (!flg_dmdAuto) {
	/* Implicit time advance of the Navier--Stokes system */
		ierr = TimeAdvance(isnapFreq, nDMD, numIters, dmdIter, flg_DMD);	CHKERRQ(ierr);
	} // Automatically set DMD
	else {
		ierr = TimeAdvanceSmart(isnapFreq, numIters, dmdIter, iDummyDMDIter, flg_DMD); CHKERRQ(ierr);
	}

	write_tecplot(out, "Field_data.dat", Soln);
	write_sym_plot_u(out, "symmetry_u.dat", Soln, JMAX + 2);

	FILE *fLOG;
	fLOG = fopen("Log.dat", "a");
	auto stop =	std::chrono::steady_clock::now();
	std::chrono::duration<double> duration = stop - start;
	fprintf(fLOG, "Total time: %f [s]\n\n\n", duration.count());
	fclose(fLOG);

	if (flg_DMD && !flg_dmdAuto) {
		ierr = PetscFree(dmdIter); CHKERRQ(ierr);
	}
	memory_deallocate();
	ierr = PetscFinalize(); CHKERRQ(ierr);
	ierr = SlepcFinalize(); CHKERRQ(ierr);

	return 0;
}

PetscErrorCode TimeAdvance(PetscInt &isnapFreq, PetscInt &nDMD, PetscInt &numIters,
		PetscInt *&dmdIter, PetscBool &flg_DMD) {
	PetscErrorCode ierr;
	int iter { 1 };
	double l2norm{};
	bool DMD_executed = false;

	FILE *residual, *solution;
	std::ofstream fML;
	fML.open("ML_dataset.csv", std::ios_base::app);

	residual = fopen("Residual.dat", "w");
	solution = fopen("solution.dat", "w");
	fprintf(solution, "%4s %14s %14s %14s\n", "iter", "dP", "dU", "dV");

	_snapshots_type snap;
	Vec vvGlobal{}, vUpdBefore{}, vUpdAfter{}, tmpDMDUpdate{};

	ierr = VecCreateSeq(PETSC_COMM_SELF, IMAX * JMAX * 3, &vvGlobal); CHKERRQ(ierr);

	for (int iDMD = 0; iDMD < nDMD + 1; iDMD++) {
		for (; iter <= dmdIter[iDMD]; iter++) {

			calc_residual(Soln, FI);
			calc_fluxJacobians(Soln, Bx, Cx, Ax, By, Cy, Ay);

			ApproximateFactorization(Soln, FI);

			fprintf(residual, "%4d %12.5G\t%.i\n", iter, l2norm, 0);

			l2norm = get_l2norm(solution, iter);

			ierr = PetscPrintf(PETSC_COMM_WORLD,
					"iter: %i dT: %3f fnorm: %6e\n", iter, dT, l2norm);
			CHKERRQ(ierr);

			ApplyBC(Soln);

			/* ----Flattening the solution into a PETSC vector----- */
			/* --------------Not including the BCs----------------- */
			for (int i = 1, index = 0; i <= IMAX; i++)
				for (int j = 1; j <= JMAX; j++)
					for (int k = 0; k < 3; k++, index++) {
						PetscScalar value = FI[i][j][k];
						ierr = VecSetValue(vvGlobal, index, value,
								INSERT_VALUES);
						CHKERRQ(ierr);
					}

			if (iter == 1 && flg_DMD) {
				ierr = initSnapsMat(vvGlobal, snap);
				CHKERRQ(ierr);
			} else if (flg_DMD && iter % isnapFreq == 0) {
				ierr = updateSolutionMatrix(vvGlobal, snap);
				CHKERRQ(ierr);
			}

			if ((iter == dmdIter[iDMD] && iter != numIters) && flg_DMD) {
#ifdef ML
				DMD_executed = true;
				if (fML.is_open()) {
					fML << "Norm1_" <<l2norm << ", ";
				}
#endif

				auto start = std::chrono::steady_clock::now();

				FILE *fLOG;
				fLOG = fopen("Log.dat", "a");
				fprintf(fLOG, "iter: %i\n", iter);


				// Recording the update before DMD
				ierr = VecDuplicate(vvGlobal, &vUpdBefore); CHKERRQ(ierr);
				ierr= VecCopy(vvGlobal, vUpdBefore); CHKERRQ(ierr);
//				ierr = printVecMATLAB("update-before" + std::to_string(dmdIter[iDMD]), "vupdBefore", vUpdBefore); CHKERRQ(ierr);

				ierr = VecNormalize(vUpdBefore, NULL); CHKERRQ(ierr);

				PetscReal dTimeStep = dT;
				DMD a_dmd(&snap.mat, dTimeStep, l2norm);

				ierr = a_dmd.applyDMDMatTrans(); CHKERRQ(ierr);

				Vec vUpdate { };
				vUpdate = a_dmd.vgetUpdate();

				// getting the dot product of the update before DMD and the DMD modes - for eigen3
				bool bEPS = true;
				for (int i = 0; i < a_dmd.iGetSVDRank(); i = i +2){
					a_dmd.dotwDMDmodes(vUpdBefore, i, bEPS);
				}

				{
					ierr = VecDuplicate(vUpdate, &tmpDMDUpdate); CHKERRQ(ierr);
					ierr = VecCopy(vUpdate, tmpDMDUpdate); CHKERRQ(ierr);
//					ierr = printVecMATLAB("update-DMD" + std::to_string(dmdIter[iDMD]), "DMDupdate", tmpDMDUpdate); CHKERRQ(ierr);

					ierr = VecNormalize(tmpDMDUpdate, NULL); CHKERRQ(ierr);
					PetscReal dot_BD{};
					ierr = VecDot(tmpDMDUpdate, vUpdBefore, &dot_BD); CHKERRQ(ierr);
					fprintf(fLOG, "<Before, DMD> = %5f\n", dot_BD);
					fclose(fLOG);

					printf("Correlation of DMD update and previous solver update: %f \n", dot_BD);
				}
				for (int i = 1, index = 0; i <= IMAX; i++)
					for (int j = 1; j <= JMAX; j++)
						for (int k = 0; k < 3; k++, index++) {
							PetscScalar value { };
							ierr = VecGetValues(vUpdate, 1, &index, &value);
							CHKERRQ(ierr);
							FI[i][j][k] = value;
							Soln[i][j][k] += value;
						}

				std::string sMessage = "Total DMD time";
				a_dmd.recordTime(start, sMessage);

			}

//			if ((iter == dmdIter[iDMD-1] + 1)){
//				ierr = VecDuplicate(vvGlobal, &vUpdAfter); CHKERRQ(ierr);
//				ierr= VecCopy(vvGlobal, vUpdAfter); CHKERRQ(ierr);
//				ierr = VecNormalize(vUpdAfter, NULL); CHKERRQ(ierr);
//
//				double dot_BD{}, dot_BA{}, dot_AD{};
//				ierr = VecDot(vUpdBefore, tmpDMDUpdate, &dot_BD); CHKERRQ(ierr);
//				ierr = VecDot(vUpdBefore, vUpdAfter, &dot_BA); CHKERRQ(ierr);
//				ierr = VecDot(vUpdAfter, tmpDMDUpdate, &dot_AD); CHKERRQ(ierr);
//
//				FILE *fLOG;
//				fLOG = fopen("Log.dat", "a");
//				fprintf(fLOG, "<Before, DMD> = %5f\n", dot_BD);
//				fprintf(fLOG, "<Before, After> = %5f\n", dot_BA);
//				fprintf(fLOG, "<After, DMD> = %5f\n", dot_AD);
//				fclose(fLOG);
//			}
#ifdef ML
			if ((iter == (dmdIter[iDMD-1] + 10) && iter != numIters) && flg_DMD) {
				if (fML.is_open()) {
					fML << "Norm2_" <<l2norm << '\n';
				}
				goto outNest;
			} else if (l2norm < 1.e-10 && DMD_executed) {
				if (fML.is_open()) {
					l2norm = get_l2norm(solution, iter);
					fML << "CONVERGED" << l2norm <<"\n";
				}
			}
#endif

			if (l2norm < 1.e-10 || iter == numIters) {
				ierr = PetscPrintf(PETSC_COMM_WORLD,
						"Converged to satisfactory point!!\n");
				CHKERRQ(ierr);
				goto outNest;
			}
		}
	}
	outNest:
	fML.close();
	fclose(residual);
	fclose(solution);
	ierr = MatDestroy(&snap.mat); CHKERRQ(ierr);
	ierr = VecDestroy(&vvGlobal); CHKERRQ(ierr);
	ierr = VecDestroy(&vUpdBefore); CHKERRQ(ierr);
	ierr = VecDestroy(&vUpdAfter); CHKERRQ(ierr);
	ierr = VecDestroy(&tmpDMDUpdate); CHKERRQ(ierr);


	return ierr;
}

PetscErrorCode TimeAdvanceSmart(PetscInt & isnapFreq, PetscInt &numIters, PetscInt *&dmdIter,
		PetscInt &iDummyDMDIter, PetscBool &flg_DMD) {
	PetscErrorCode ierr;
	int iter { 1 };
	double l2norm{};

	FILE *residual, *solution;
	residual = fopen("Residual.dat", "w");
	solution = fopen("solution.dat", "w");
	fprintf(solution, "%4s %14s %14s %14s\n", "iter", "dP", "dU", "dV");

	_snapshots_type snap;
	Vec vvGlobal{}, vUpdBefore{};
	PetscReal dTimeStep = dT;

	ierr = VecCreateSeq(PETSC_COMM_SELF, IMAX * JMAX * 3, &vvGlobal);
	CHKERRQ(ierr);
	ierr = VecCreateSeq(PETSC_COMM_SELF, IMAX * JMAX * 3, &vUpdBefore);
	CHKERRQ(ierr);

	int iSCP{}; //Slope Check Period
	// variables for checking the slope of the residual
	std::vector<double> LRx; // x values
	std::deque<double> dResWin { }; // y values
	PetscScalar slope, intercept;
	PetscScalar slope_old = 1, slope_ratio = 0;

	/* Implicit time advance of the Navier--Stokes system */
	for (; iter <= numIters; iter++) {

		calc_residual(Soln, FI);
		calc_fluxJacobians(Soln, Bx, Cx, Ax, By, Cy, Ay);

		ApproximateFactorization(Soln, FI);
		l2norm = get_l2norm(solution, iter);

		if (iter == iDummyDMDIter) {

			DMD DMDTest(&snap.mat, dTimeStep, l2norm);
			ierr = DMDTest.DummyDMD(); CHKERRQ(ierr);
			iSCP = DMDTest.iGetDominantPeriod();
			ierr = PetscPrintf(PETSC_COMM_WORLD,
							"\nFor Width: %3f, Height: %3f, the oscillation period is computed as: %3d\n",
							Width, Height, iSCP); CHKERRQ(ierr);

			for (int j = 0; j < iSCP; j++) {
				LRx.push_back(j);
			}
		}

		if (iter >= iDummyDMDIter)
			dResWin.push_back(l2norm);

		// Handling the residual, data to decide when to apply the relaxation //
		if (iter >= iDummyDMDIter + iSCP && abs(iSCP) > 1) {
			dResWin.pop_front();

			Linear_Regression(LRx, dResWin, slope, intercept);

			if (iter % (iSCP) == 0) {
				slope_ratio = slope / slope_old;
				slope_old = slope;
			} else {
				slope_ratio = -1;
			}

			fprintf(residual, "%4d %12.5G\t %.8G\t %8.5G\n", iter, l2norm, slope, slope_ratio);

		} else {
			fprintf(residual, "%4d %12.5G\t%.i\n", iter, l2norm, 0);
		}

		ierr = PetscPrintf(PETSC_COMM_WORLD, "iter: %i dT: %3f fnorm: %6e\n",
				iter, dT, l2norm);
		CHKERRQ(ierr);

		ApplyBC(Soln);


		/* --------Not including the BCs-------- */
		for (int i = 1, index = 0; i <= IMAX; i++)
			for (int j = 1; j <= JMAX; j++)
				for (int k = 0; k < 3; k++, index++) {
					PetscScalar value = FI[i][j][k];
					ierr = VecSetValue(vvGlobal, index, value, INSERT_VALUES);
					CHKERRQ(ierr);
				}

		// Initializing the snapshot matrix in iter = 1, and updating it in other iterations
		if (iter == 1 && flg_DMD) {
			ierr = initSnapsMat(vvGlobal, snap);
			CHKERRQ(ierr);
		} else if (flg_DMD && iter % isnapFreq == 0) {
			ierr = updateSolutionMatrix(vvGlobal, snap);
			CHKERRQ(ierr);
		}

		if (abs(slope_ratio - 1) <= 0.0011 && flg_DMD) {

			auto start = std::chrono::steady_clock::now();

			FILE *fLOG;
			fLOG = fopen("Log.dat", "a");
			fprintf(fLOG, "iter: %i\n", iter);
			fclose(fLOG);

			// Recording the update before DMD
			ierr= VecCopy(vvGlobal, vUpdBefore); CHKERRQ(ierr);
			ierr = VecNormalize(vUpdBefore, NULL); CHKERRQ(ierr);

			DMD a_dmd(&snap.mat, dTimeStep, l2norm);
			ierr = a_dmd.applyDMDMatTrans(); CHKERRQ(ierr);

			Vec vUpdate { };
			vUpdate = a_dmd.vgetUpdate();

			Vec tmpDMDUpdate{};
			ierr = VecDuplicate(vUpdate, &tmpDMDUpdate); CHKERRQ(ierr);
			ierr = VecCopy(vUpdate, tmpDMDUpdate); CHKERRQ(ierr);
			ierr = VecNormalize(tmpDMDUpdate, NULL); CHKERRQ(ierr);
			PetscReal dot_BD{};
			ierr = VecDot(tmpDMDUpdate, vUpdBefore, &dot_BD); CHKERRQ(ierr);

			printf("Correlation of DMD update and previous solver update: %f \n", dot_BD);
			if(dot_BD > CORETHRESH){
				printf("Applying the computed update.\n\n");

			for (int i = 1, index = 0; i <= IMAX; i++)
				for (int j = 1; j <= JMAX; j++)
					for (int k = 0; k < 3; k++, index++) {
						PetscScalar value { };
						ierr = VecGetValues(vUpdate, 1, &index, &value);
						CHKERRQ(ierr);
						FI[i][j][k] = value;
						Soln[i][j][k] += value;
					}
			} else {
				printf("DMD update discarded....\n");
			}
			std::string sMessage = "Total DMD time";
			a_dmd.recordTime(start, sMessage);
		}

		if (l2norm < 1.e-10 || iter == numIters) {
			ierr = PetscPrintf(PETSC_COMM_WORLD,
					"Converged to satisfactory point!!\n");
			CHKERRQ(ierr);
			goto outNest;
		}
	}
	outNest:
	fclose(residual);
	fclose(solution);
	ierr = MatDestroy(&snap.mat); CHKERRQ(ierr);
	ierr = VecDestroy(&vvGlobal); CHKERRQ(ierr);
	ierr = VecDestroy(&vUpdBefore); CHKERRQ(ierr);

	return ierr;
}



/* calculating norm */
double get_l2norm(FILE* solution, int iter) {
	L2NORM l2norm{};

	for (int i = 1; i < IMAX + 2; i++) {
		for (int j = 1; j < JMAX + 2; j++) {
			ERROR error{};

			error._1 = FI[i][j][0];
			error._2 = FI[i][j][1];
			error._3 = FI[i][j][2];

			l2norm._1 += error._1 * error._1;
			l2norm._2 += error._2 * error._2;
			l2norm._3 += error._3 * error._3;
		}
	}

	l2norm._1 = sqrt(l2norm._1 / (IMAX * JMAX));
	l2norm._2 = sqrt(l2norm._2 / (IMAX * JMAX));
	l2norm._3 = sqrt(l2norm._3 / (IMAX * JMAX));

	double MaxChange = MAX(l2norm._1, l2norm._2);
	MaxChange = MAX(MaxChange, l2norm._3);

	fprintf(solution, "%4d %14G %14G %14G\n",
		iter, l2norm._1, l2norm._2, l2norm._3);

	return MaxChange;
}

PetscErrorCode readOpts(PetscReal &dT, PetscInt &isnapFreq, PetscInt &max_iter_total,
		PetscBool &flg_DMD, PetscBool &flg_dmdAuto, PetscInt &nDMD,
		PetscInt *&dmdIter, PetscInt &iDummyDMDIter) {
	PetscErrorCode ierr;
	PetscBool flg;

	ierr = PetscOptionsGetReal(NULL, NULL, "-dt", &dT, &flg);
	CHKERRQ(ierr);
	if (!flg)
		ierr = PetscErrorPrintf("Missing -dt option!\n");	CHKERRQ(ierr);

	ierr = PetscOptionsGetInt(NULL, PETSC_NULL, "-snapshot_frequency",
			&isnapFreq, &flg);	CHKERRQ(ierr);

	ierr = PetscOptionsGetInt(NULL, PETSC_NULL, "-max_iter_total",
			&max_iter_total, &flg);	CHKERRQ(ierr);
	if (!flg)
		PetscErrorPrintf("Missing -max_iter_total flag!\n");

	/* ------------------- DMD Options ------------- */

	ierr = PetscOptionsHasName(NULL, PETSC_NULL, "-DMD", &flg_DMD);	CHKERRQ(ierr);

	if (flg_DMD) {
		ierr = PetscOptionsHasName(NULL, PETSC_NULL, "-DMD_auto_config",
				&flg_dmdAuto);	CHKERRQ(ierr);
		if (flg_dmdAuto) {
			ierr = PetscPrintf(PETSC_COMM_WORLD, "Auto configure DMD.\n"); CHKERRQ(ierr);
		} else {
			ierr = PetscPrintf(PETSC_COMM_WORLD,
					"Manually configuring DMD.\nConfig should be provided\n"); CHKERRQ(ierr);
		}
	}

	if (flg_DMD && !flg_dmdAuto) {
		ierr = PetscOptionsGetInt(NULL, NULL, "-DMD_nits", &nDMD, NULL); CHKERRQ(ierr);
		ierr = PetscMalloc1(nDMD + 1, &dmdIter);
		CHKERRQ(ierr);
		dmdIter[nDMD] = max_iter_total;

		ierr = PetscOptionsGetIntArray(PETSC_NULL, PETSC_NULL, "-DMD_its",
				dmdIter, &nDMD, &flg); CHKERRQ(ierr);
		if (flg) {
			if (nDMD < 1) {
				PetscErrorPrintf("Incorrect argument for -DMD_its\n");
				exit(1);
			}
		}
		std::sort(dmdIter, dmdIter + nDMD + 1);
	} else if (flg_DMD && flg_dmdAuto) {

		ierr = PetscOptionsGetInt(NULL, PETSC_NULL, "-DummyDMDIter",
				&iDummyDMDIter, &flg); CHKERRQ(ierr);
	if (!flg){
		PetscErrorPrintf("Missing -DummyDMDIter flag!\n");
		iDummyDMDIter = 250;
	}

	nDMD = 0;
	ierr = PetscMalloc(nDMD + 1, &dmdIter); CHKERRQ(ierr);
	dmdIter[nDMD] = max_iter_total;
	}
	return ierr;
}

/* ----Setting up the snapshot matrix in a Row-Major form---- */
PetscErrorCode initSnapsMat(Vec& vec, _snapshots_type& snap) {
	PetscErrorCode ierr;
	PetscBool flg;

	ierr = PetscOptionsGetInt(NULL, PETSC_NULL, "-svd_ncv", &snap.iNumRows, &flg);CHKERRQ(ierr);
		if (!flg){
			PetscErrorPrintf("Missing -svd_ncv flag!\n");
			exit(2);
		}

	// Getting the size of each snapshot data
	ierr = VecGetSize(vec, &snap.iNumCols);CHKERRQ(ierr);

	for (int i = 0; i < snap.iNumCols; i++) {
		snap.col_index.push_back(i);
		snap.vValues.push_back(0.0);
	}

	snap.last_row_index.push_back(snap.iNumRows - 1);

	for (int j = snap.iNumRows - 1; j > 0; j--) {
		snap.old_row_indices.push_back(j);
		snap.new_row_indices.push_back(j - 1);
		assert(((j-1)>=0) && "index for updating PCA matrix is out of range");
	}

	for (int j = 0; j < snap.iNumRows; j++) {
		snap.row_indices.push_back(j);
		for (int i = 0; i < snap.iNumCols; i++) {
			snap.mValues.push_back(0.0);
		}
	}
#ifdef SNAPS_SUB_MEAN
	PetscReal sum{};
	ierr = VecSum(vec, &sum);CHKERRQ(ierr);
	ierr = VecShift(vec, -sum / (double) snap.iNumCols);CHKERRQ(ierr);
#endif
#ifdef SNAPS_VEC_NORMALIZE
	ierr = VecNormalize(vec, NULL); CHKERRQ(ierr);
#endif
	// getting the flattened data from the solution vector
	ierr = VecGetValues(vec, snap.iNumCols, snap.col_index.data(), snap.vValues.data());CHKERRQ(ierr);

	// Setting up the each solution in each row (short-fat matrix instead of tall-skinny)
	ierr = MatDestroy(&snap.mat);	CHKERRQ(ierr);
	ierr = MatCreate(MPI_COMM_WORLD, &snap.mat);	CHKERRQ(ierr);
	ierr = MatSetSizes(snap.mat, PETSC_DECIDE, PETSC_DECIDE, snap.iNumRows,snap.iNumCols);	CHKERRQ(ierr);
	ierr = MatSetType(snap.mat, MATAIJ); CHKERRQ(ierr);
	ierr = MatSetUp(snap.mat);	CHKERRQ(ierr);

	 //setting the last column of the snapshots matrix
	ierr = MatSetValues(snap.mat, 1, snap.last_row_index.data(),
			snap.iNumCols, snap.col_index.data(), snap.vValues.data(),
			INSERT_VALUES);CHKERRQ(ierr);

	ierr = MatAssemblyBegin(snap.mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(snap.mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	return ierr;
}

/*  * Updating the snapshot matrix by inserting new rows (short-fat matrix)
 * This is a row-major d ataset, each solution vector is defined in each row
 */
PetscErrorCode updateSolutionMatrix(Vec& vec, _snapshots_type& snap) {
	PetscErrorCode ierr;

	ierr = MatGetValues(snap.mat, snap.iNumRows - 1, snap.old_row_indices.data(),
			snap.iNumCols, snap.col_index.data(),
			snap.mValues.data()); CHKERRQ(ierr);

	ierr = MatSetValues(snap.mat, snap.iNumRows - 1, snap.new_row_indices.data(),
			snap.iNumCols, snap.col_index.data(),
			snap.mValues.data(), INSERT_VALUES); CHKERRQ(ierr);
#ifdef SNAPS_SUB_MEAN
	PetscReal sum{};
	ierr = VecSum(vec, &sum);CHKERRQ(ierr);
	ierr = VecShift(vec, -sum / (double) snap.iNumCols);CHKERRQ(ierr);
#endif
#ifdef SNAPS_VEC_NORMALIZE
	ierr = VecNormalize(vec, NULL); CHKERRQ(ierr);
#endif
	//getting the serialized data from the solution vector
	ierr = VecGetValues(vec, snap.iNumCols, snap.col_index.data(), snap.vValues.data()); CHKERRQ(ierr);

	/* Setting the last column of the snapshots Matrix */
	ierr = MatSetValues(snap.mat, 1, snap.last_row_index.data(),
			snap.iNumCols, snap.col_index.data(), snap.vValues.data(),
			INSERT_VALUES); CHKERRQ(ierr);

	ierr = MatAssemblyBegin(snap.mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(snap.mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	return ierr;
}

PetscErrorCode printVecMATLAB(std::string sFileName,
		std::string sVectorName, Vec V) {
	PetscErrorCode ierr;

	std::string sName = sFileName + ".m";
	PetscViewer viewer;
	PetscObjectSetName((PetscObject) V, sVectorName.c_str());
	PetscViewerASCIIOpen(PETSC_COMM_WORLD, sName.c_str(), &viewer);
	PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
	ierr = VecView(V, viewer);
	CHKERRQ(ierr);
	PetscViewerPopFormat(viewer);
	PetscViewerDestroy(&viewer);

	return ierr;
}

PetscErrorCode printMatMATLAB(std::string sFilename,
		std::string sMatrixName, Mat A) {
	PetscErrorCode ierr;

	std::string sName = sFilename + ".m";
	PetscViewer viewer;
	PetscObjectSetName((PetscObject) A, sMatrixName.c_str());
	PetscViewerASCIIOpen(PETSC_COMM_WORLD, sName.c_str(), &viewer);
	PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
	ierr = MatView(A, viewer);	CHKERRQ(ierr);
	PetscViewerPopFormat(viewer);
	PetscViewerDestroy(&viewer);

	return ierr;
}

PetscErrorCode printVecPYTHON(std::string sFileName,
		std::string sVectorName, Vec V) {
	PetscErrorCode ierr;

	std::string sName = sFileName + ".csv";
	PetscViewer viewer;
	PetscObjectSetName((PetscObject) V, sVectorName.c_str());
	PetscViewerASCIIOpen(PETSC_COMM_WORLD, sName.c_str(), &viewer);
	PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_PYTHON);
	ierr = VecView(V, viewer);
	CHKERRQ(ierr);
	PetscViewerPopFormat(viewer);
	PetscViewerDestroy(&viewer);

	return ierr;
}

PetscErrorCode printMatPYTHON(std::string sFilename,
		std::string sMatrixName, Mat A) {
	PetscErrorCode ierr;

	std::string sName = sFilename + ".csv";
	PetscViewer viewer;
	PetscObjectSetName((PetscObject) A, sMatrixName.c_str());
	PetscViewerASCIIOpen(PETSC_COMM_WORLD, sName.c_str(), &viewer);
	PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_PYTHON);
	ierr = MatView(A, viewer);
	CHKERRQ(ierr);
	PetscViewerPopFormat(viewer);
	PetscViewerDestroy(&viewer);

	return ierr;
}

template <class T , class M>
void Linear_Regression(std::vector<T> indep_var, std::deque<T> dep_var, M &a_1, M &a_2)
{

	int n = indep_var.size();
	int n_test = dep_var.size();
	assert(n==n_test && "The size of dependent and independent variables are not equal, so we cannot do regression\n");

	for (auto it = dep_var.begin(); it != dep_var.end(); ++it)
		*it = log(*it);

	T Sxx{}, Sx{}, S{static_cast<T>(n)};
	T Sxy{}, Sy{}, Delta{};
	for (int i=0; i < n; i++){
		Sxx += indep_var[i] * indep_var[i];
		Sx += indep_var[i];
		Sxy += indep_var[i] * dep_var[i];
		Sy += dep_var[i];
	}

	Delta = S*Sxx - Sx*Sx;

	a_1 = (S*Sxy - Sx*Sy)/Delta;
	a_2 = (Sxx*Sy - Sx*Sxy)/Delta;

	a_1 = exp(a_1);

}




