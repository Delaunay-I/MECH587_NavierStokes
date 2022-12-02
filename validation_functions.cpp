#include "functions.h"
// #include <math.h>

/* Calculating flux integral at the center of CV */
void exact_fluxIntegral(double*** FI) {
	for (int i = 0; i < IMAX; i++) {
		double X = (i + 0.5) * dX;


		/* parameters of exact FluxIntegral*/
		double Cx = cos(PI * X);
		double Sx = sin(PI * X);
		double C2x = cos(2 * PI * X);
		double S2x = sin(2 * PI * X);


		for (int j = 0; j < JMAX; j++) {
			double Y = (j + 0.5) * dY;

			/* parameters of exact FluxIntegral*/
			double Cy = cos(PI * Y);
			double Sy = sin(PI * Y);
			double C2y = cos(2 * PI * Y);
			double S2y = sin(2 * PI * Y);

			/* calculating the exact Flux Integral */
			FI[i][j][0] = -(PI / Beta) * (u0 * Cx * S2y + my_v0 * S2x * Cy);

			FI[i][j][1] = P0 * PI * Sx * Cy - u0 * u0 * PI * S2x * S2y * S2y
				- u0 * my_v0 * PI * Sx * S2x * (Cy * S2y + 2 * C2y * Sy)
				- (u0 * 5.0 * PI * PI * Sx * S2y) / Re;

			FI[i][j][2] = P0 * PI * Cx * Sy - my_v0 * my_v0 * PI * S2x * S2x * S2y
				- u0 * my_v0 * PI * Sy * S2y * (Cx * S2x + 2 * C2x * Sx)
				- (my_v0 * 5.0 * PI * PI * S2x * Sy) / Re;
		}
	}
}


/* calculating flux jacobian and flux integral for testing flux jacobian correctness */
void calc_fluxJacobians(double*** Soln, double*** dSoln, double*** RHS) {

	/*double FFJ1[IMAX + 2][JMAX + 2][3][3]{}, FFJ2[IMAX + 2][JMAX + 2][3][3]{};
	double GFJ1[IMAX + 2][JMAX + 2][3][3]{}, GFJ2[IMAX + 2][JMAX + 2][3][3]{};*/


	double**** FFJ1{}, **** FFJ2{};
	double**** GFJ1{}, **** GFJ2{};

	mem_alloc_fluxJacobian(&FFJ1, &FFJ2, &GFJ1, &GFJ2);

	for (int j = 1; j < JMAX + 1; j++) {
		for (int i = 1; i < IMAX + 2; i++) {
			int I_ = i - 1;
			/* Defining Jacobian type 1: F(i+0.5,j)/U(i,j) and F(i-0.5,j)/U(i-1,j) */
			FFJ1[i][j][0][0] = 0;	FFJ1[i][j][0][1] = 1.0 / (2.0 * Beta);											FFJ1[i][j][0][2] = 0;
			FFJ1[i][j][1][0] = 0.5;	FFJ1[i][j][1][1] = (Soln[I_][j][1] + Soln[I_ + 1][j][1]) / 2.0 + 1.0 / (dX * Re); FFJ1[i][j][1][2] = 0;
			FFJ1[i][j][2][0] = 0;	FFJ1[i][j][2][1] = (Soln[I_][j][2] + Soln[I_ + 1][j][2]) / 4.0;				    FFJ1[i][j][2][2] = (Soln[I_][j][1] + Soln[I_ + 1][j][1]) / 4.0 + 1.0 / (dX * Re);
			/* Defining Jacobian type 2: F(i+0.5,j)/U(i+1,j) and F(i-0.5,j)/U(i-1,j) */
			FFJ2[i][j][0][0] = 0;	FFJ2[i][j][0][1] = 1.0 / (2.0 * Beta);										    FFJ2[i][j][0][2] = 0;
			FFJ2[i][j][1][0] = 0.5;	FFJ2[i][j][1][1] = (Soln[I_][j][1] + Soln[I_ + 1][j][1]) / 2.0 - 1.0 / (dX * Re); FFJ2[i][j][1][2] = 0;
			FFJ2[i][j][2][0] = 0;	FFJ2[i][j][2][1] = (Soln[I_][j][2] + Soln[I_ + 1][j][2]) / 4.0;				    FFJ2[i][j][2][2] = (Soln[I_][j][1] + Soln[I_ + 1][j][1]) / 4.0 - 1.0 / (dX * Re);
		}
	}

	for (int i = 1; i < IMAX + 1; i++) {
		for (int j = 1; j < JMAX + 2; j++) {
			int J = j - 1;
			/* Defining Jacobian type 1: F(i+0.5,j)/U(i,j) */
			GFJ1[i][j][0][0] = 0;		GFJ1[i][j][0][1] = 0;															GFJ1[i][j][0][2] = 1.0 / (2.0 * Beta);
			GFJ1[i][j][1][0] = 0;		GFJ1[i][j][1][1] = (Soln[i][J][2] + Soln[i][J + 1][2]) / 4.0 + 1.0 / (dY * Re); GFJ1[i][j][1][2] = (Soln[i][J][1] + Soln[i][J + 1][1]) / 4.0;
			GFJ1[i][j][2][0] = 0.5;		GFJ1[i][j][2][1] = 0;															GFJ1[i][j][2][2] = (Soln[i][J][2] + Soln[i][J + 1][2]) / 2.0 + 1.0 / (dY * Re);
			/* Defining Jacobian type 2: F(i+0.5,j)/U(i+1,j) */
			GFJ2[i][j][0][0] = 0;		GFJ2[i][j][0][1] = 0;															GFJ2[i][j][0][2] = 1.0 / (2.0 * Beta);
			GFJ2[i][j][1][0] = 0;		GFJ2[i][j][1][1] = (Soln[i][J][2] + Soln[i][J + 1][2]) / 4.0 - 1.0 / (dY * Re); GFJ2[i][j][1][2] = (Soln[i][J][1] + Soln[i][J + 1][1]) / 4.0;
			GFJ2[i][j][2][0] = 0.5;		GFJ2[i][j][2][1] = 0;															GFJ2[i][j][2][2] = (Soln[i][J][2] + Soln[i][J + 1][2]) / 2.0 - 1.0 / (dY * Re);
		}
	}



	for (int i = 0; i < IMAX; i++) {
		for (int j = 0; j < JMAX; j++) {
			double Ax{}, Bx{}, Cx{}; //considers the multiplication of delta_U
			double Ay{}, By{}, Cy{}; //this one too			

			for (int k = 0; k < 3; k++) {
				int I_ = i + 2;
				int Iu = i + 1;
				//int k = 0;
				Bx = (FFJ1[I_][j + 1][k][0] * dSoln[Iu][j + 1][0] - FFJ2[I_ - 1][j + 1][k][0] * dSoln[Iu][j + 1][0]) / dX;
				Bx += (FFJ1[I_][j + 1][k][1] * dSoln[Iu][j + 1][1] - FFJ2[I_ - 1][j + 1][k][1] * dSoln[Iu][j + 1][1]) / dX;
				Bx += (FFJ1[I_][j + 1][k][2] * dSoln[Iu][j + 1][2] - FFJ2[I_ - 1][j + 1][k][2] * dSoln[Iu][j + 1][2]) / dX;

				Cx = (FFJ2[I_][j + 1][k][0] * dSoln[Iu + 1][j + 1][0]) / dX;
				Cx += (FFJ2[I_][j + 1][k][1] * dSoln[Iu + 1][j + 1][1]) / dX;
				Cx += (FFJ2[I_][j + 1][k][2] * dSoln[Iu + 1][j + 1][2]) / dX;

				Ax = -(FFJ1[I_ - 1][j + 1][k][0] * dSoln[Iu - 1][j + 1][0]) / dX;
				Ax += -(FFJ1[I_ - 1][j + 1][k][1] * dSoln[Iu - 1][j + 1][1]) / dX;
				Ax += -(FFJ1[I_ - 1][j + 1][k][2] * dSoln[Iu - 1][j + 1][2]) / dX;

				I_ = i + 1;
				int J = j + 2;
				int Ju = j + 1;

				By = (GFJ1[I_][J][k][0] * dSoln[I_][Ju][0] - GFJ2[I_][J - 1][k][0] * dSoln[I_][Ju][0]) / dY;
				By += (GFJ1[I_][J][k][1] * dSoln[I_][Ju][1] - GFJ2[I_][J - 1][k][1] * dSoln[I_][Ju][1]) / dY;
				By += (GFJ1[I_][J][k][2] * dSoln[I_][Ju][2] - GFJ2[I_][J - 1][k][2] * dSoln[I_][Ju][2]) / dY;

				Cy = (GFJ2[I_][J][k][0] * dSoln[I_][Ju + 1][0]) / dY;
				Cy += (GFJ2[I_][J][k][1] * dSoln[I_][Ju + 1][1]) / dY;
				Cy += (GFJ2[I_][J][k][2] * dSoln[I_][Ju + 1][2]) / dY;


				Ay = -(GFJ1[I_][J - 1][k][0] * dSoln[I_][Ju - 1][0]) / dY;
				Ay += -(GFJ1[I_][J - 1][k][1] * dSoln[I_][Ju - 1][1]) / dY;
				Ay += -(GFJ1[I_][J - 1][k][2] * dSoln[I_][Ju - 1][2]) / dY;


				RHS[i][j][k] = Bx + By + Cx + Ax + Cy + Ay;

			}
		}
	}
	mem_dealloc_fluxJacobian(&FFJ1, &FFJ2, &GFJ1, &GFJ2);
}

/* calculating flux integral with the given solution changes for testing */
void calc_exact_LHS(double*** FI_old, double*** FI_new, double*** LHS) {
	for (int i = 0; i < IMAX; i++) {
		for (int j = 0; j < JMAX; j++) {
			LHS[i][j][0] = FI_new[i][j][0] - FI_old[i][j][0];
			LHS[i][j][1] = FI_new[i][j][1] - FI_old[i][j][1];
			LHS[i][j][2] = FI_new[i][j][2] - FI_old[i][j][2];
		}
	}
}

void soln_advance(double*** Soln, double*** dSoln) {
	for (int i = 0; i < IMAX + 2; i++) {
		for (int j = 0; j < JMAX + 2; j++) {
			Soln[i][j][0] += dSoln[i][j][0];
			Soln[i][j][1] += dSoln[i][j][1];
			Soln[i][j][2] += dSoln[i][j][2];
		}
	}
}