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
	std::vector<PetscInt> row_index, column_indices;
	std::vector<PetscInt> old_column_indices, new_column_indices;
	std::vector<PetscInt> last_column_index; // REMOVE THIS
	std::vector<PetscScalar> vValues; // REMOVE THIS
	std::vector<PetscScalar> mValues; // REMOVE THIS //a placeholder for values - for updating svd results in SVD.mat
	};


/* DMD added Functions */
PetscErrorCode initSnapsMat(Vec& vec, _snapshots_type& snap);
PetscErrorCode updateSolutionMatrix(Vec& vec, _snapshots_type& snap);

/* declaring global variables */
double*** Soln{}, *** dSoln{}, ***Soln2{};
double*** FI{};

/* 4D arrays */
double**** Bx{}, **** Cx{}, **** Ax{};
double**** By{}, **** Cy{}, **** Ay{};

/* -----------Initializing the solution-------- */
void initialize(double*** Soln, double*** dSoln) {
	for (int i = 0; i < IMAX + 2; i++) {
		double X = (i - 0.5) * dX;
		for (int j = 0; j < JMAX + 2; j++) {
			double Y = (j - 0.5) * dY;
			Soln[i][j][0] = P0 * cos(PI * X) * cos(PI * Y);
			Soln[i][j][1] = u0 * sin(PI * X) * sin(2 * PI * Y);
			Soln[i][j][2] = v0 * sin(2 * PI * X) * sin(PI * Y);
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
	PetscMPIInt size;

	ierr = SlepcInitialize(&argc, &argv, (char*) 0, (char*) 0); CHKERRQ(ierr);
	if (ierr)
		return ierr;

	ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size); CHKERRQ(ierr);

	FILE* out{};
	FILE* residual, * solution;

	memory_allocation();

	/* initializing P, u, and v with the initial conditions */
	initialize(Soln, dSoln);
	
	int iter{};
	double MaxChange{};
	residual = fopen("Residuals.dat", "w");
	solution = fopen("solution.dat", "w");
	fprintf(solution, "%4s %14s %14s %14s\n", "iter", "dP", "dU", "dV");

	_snapshots_type snap;
	Vec vvGlobal;

	/* Implicit time advance of the Navier--Stokes system */
	do
	{
		iter++;
		calc_residual(Soln, FI);
		calc_fluxJacobians(Soln, Bx, Cx, Ax, By, Cy, Ay);

		MaxChange = ApproximateFactorization(Soln, FI);
		fprintf(residual, "%4d %12.5G\n", iter, MaxChange);

		get_l2norm(solution, iter);

		ApplyBC(Soln);

		ierr = VecCreateSeq(PETSC_COMM_SELF, IMAX*JMAX*3, &vvGlobal); CHKERRQ(ierr);
		/* --------Not including the BCs-------- */
		for (int i = 1; i <= IMAX; i++)
			for (int j = 1; j <= JMAX; j++)
				for (int k = 0; k < 3; k++) {
					PetscScalar value = FI[i][j][k];
					ierr = VecSetValue(vvGlobal, i - 1, value, INSERT_VALUES); CHKERRQ(ierr);
				}
		if (iter == 1)
			ierr = initSnapsMat(vvGlobal, snap); CHKERRQ(ierr);

	} while (iter < MAX_ITER && MaxChange > Tol);

	fclose(residual);
	fclose(solution);


	write_tecplot(out, "Field_data.dat", Soln);

	write_sym_plot_u(out, "symmetry_u.dat", Soln, JMAX + 2);


	memory_deallocate();
	ierr = SlepcFinalize(); CHKERRQ(ierr);

	return 0;
}

/* calculating norm */
void get_l2norm(FILE* solution, int iter) {


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

	fprintf(solution, "%4d %14G %14G %14G\n",
		iter, l2norm._1, l2norm._2, l2norm._3);
}


PetscErrorCode initSnapsMat(Vec& vec, _snapshots_type& snap) {
	PetscErrorCode ierr;
	PetscBool flg;

	ierr = PetscOptionsGetInt(NULL, PETSC_NULL, "-svd_ncv", &snap.iNumCols, &flg);CHKERRQ(ierr);
		if (!flg)
			PetscErrorPrintf("Missing -svd_ncv flag!\n");

	// Getting the number of rows of each snapshot data
	ierr = VecGetSize(vec, &snap.iNumRows);CHKERRQ(ierr);

	for (int i = 0; i < snap.iNumRows; i++) {
		snap.row_index.push_back(i);
		snap.vValues.push_back(0.0);
	}

	snap.last_column_index.push_back(snap.iNumCols - 1);

	for (int j = snap.iNumCols - 1; j > 0; j--) {
		snap.old_column_indices.push_back(j);
		snap.new_column_indices.push_back(j - 1);
		assert(((j-1)>=0) && "index for updating PCA matrix is out of range");
	}

	for (int j = 0; j < snap.iNumCols; j++) {
		snap.column_indices.push_back(j);
		for (int i = 0; i < snap.iNumRows; i++) {
			snap.mValues.push_back(0.0);
		}
	}

	//getting the serialized data from the solution vector
	ierr = VecGetValues(vec, snap.iNumRows, snap.row_index.data(), snap.vValues.data());CHKERRQ(ierr);

	ierr = MatDestroy(&snap.mat);	CHKERRQ(ierr);
	ierr = MatCreate(MPI_COMM_WORLD, &snap.mat);	CHKERRQ(ierr);
	ierr = MatSetSizes(snap.mat, PETSC_DECIDE, PETSC_DECIDE, snap.iNumRows,snap.iNumCols);	CHKERRQ(ierr);
	ierr = MatSetType(snap.mat, MATAIJ); CHKERRQ(ierr);
	ierr = MatSetUp(snap.mat);	CHKERRQ(ierr);

	 //setting the last column of the snapshots matrix
	ierr = MatSetValues(snap.mat, snap.iNumRows, snap.row_index.data(),
			1, snap.last_column_index.data(), snap.vValues.data(),
			INSERT_VALUES);CHKERRQ(ierr);

	ierr = MatAssemblyBegin(snap.mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(snap.mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	return ierr;
}


PetscErrorCode updateSolutionMatrix(Vec& vec, _snapshots_type& snap) {
	PetscErrorCode ierr;

	ierr = MatGetValues(snap.mat, snap.iNumRows, snap.row_index.data(),
			snap.iNumCols - 1, snap.old_column_indices.data(),
			snap.mValues.data()); CHKERRQ(ierr);

	ierr = MatSetValues(snap.mat, snap.iNumRows, snap.row_index.data(),
			snap.iNumCols - 1, snap.new_column_indices.data(),
			snap.mValues.data(), INSERT_VALUES); CHKERRQ(ierr);

	//getting the serialized data from the solution vector
	ierr = VecGetValues(vec, snap.iNumRows, snap.row_index.data(), snap.vValues.data()); CHKERRQ(ierr);

	/* Setting the last column of the snapshots Matrix */
	ierr = MatSetValues(snap.mat, snap.iNumRows, snap.row_index.data(),
			1, snap.last_column_index.data(), snap.vValues.data(),
			INSERT_VALUES); CHKERRQ(ierr);

	ierr = MatAssemblyBegin(snap.mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(snap.mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	return ierr;
}


