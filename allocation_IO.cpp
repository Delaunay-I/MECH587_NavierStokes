#include "functions.h"


void mem_alloc_residual(double**** Fx, double**** Fy, double**** FIx, double**** FIy) {

	/* ------ IMAX +2 * JMAX + 2------- */
	*Fx = new double** [IMAX + 2]{};
	*Fy = new double** [IMAX + 2]{};
	for (int i = 0; i < IMAX + 2; i++) {
		(*Fx)[i] = new double* [JMAX + 2];
		(*Fy)[i] = new double* [JMAX + 2];
		for (int j = 0; j < JMAX + 2; j++) {
			(*Fx)[i][j] = new double[3]{};
			(*Fy)[i][j] = new double[3]{};
		}
	}
	/*  ------ IMAX * JMAX -------  */
	*FIx = { new double** [IMAX] {} };
	*FIy = { new double** [IMAX] {} };
	for (int i = 0; i < IMAX; i++) {
		(*FIx)[i] = new double* [JMAX];
		(*FIy)[i] = new double* [JMAX];
		for (int j = 0; j < JMAX; j++) {
			(*FIx)[i][j] = new double[3]{};
			(*FIy)[i][j] = new double[3]{};
		}
	}
}

void mem_dealloc_residual(double**** Fx, double**** Fy, double**** FIx, double**** FIy) {
	/* ------ IMAX + 2 * JMAX + 2------- */
	for (int i = 0; i < IMAX + 2; i++) {
		for (int j = 0; j < JMAX + 2; j++) {
			delete[](*Fx)[i][j];
			delete[](*Fy)[i][j];
		}
		delete[](*Fx)[i];
		delete[](*Fy)[i];
	}
	delete[] * Fx;
	delete[] * Fy;

	/* ------ IMAX * JMAX ------- */
	for (int i = 0; i < IMAX; i++) {
		for (int j = 0; j < JMAX; j++) {
			delete[](*FIx)[i][j];
			delete[](*FIy)[i][j];
		}
		delete[](*FIx)[i];
		delete[](*FIy)[i];
	}
	delete[] * FIx;
	delete[] * FIy;
}

void mem_alloc_fluxJacobian(double***** FFJ1, double***** FFJ2, double***** GFJ1, double***** GFJ2) {

	(*FFJ1) = new double*** [IMAX + 2]{};
	(*FFJ2) = new double*** [IMAX + 2]{};
	(*GFJ1) = new double*** [IMAX + 2]{};
	(*GFJ2) = new double*** [IMAX + 2]{};
	

	for (int i = 0; i < IMAX + 2; i++) {
		(*FFJ1)[i] = new double** [JMAX + 2]{};
		(*FFJ2)[i] = new double** [JMAX + 2]{};
		(*GFJ1)[i] = new double** [JMAX + 2]{};
		(*GFJ2)[i] = new double** [JMAX + 2]{};
		

		for (int j = 0; j < JMAX + 2; j++) {
			(*FFJ1)[i][j] = new double* [3]{};
			(*FFJ2)[i][j] = new double* [3]{};
			(*GFJ1)[i][j] = new double* [3]{};
			(*GFJ2)[i][j] = new double* [3]{};
			

			for (int k = 0; k < 3; k++) {
				(*FFJ1)[i][j][k] = new double[3]{};
				(*FFJ2)[i][j][k] = new double[3]{};
				(*GFJ1)[i][j][k] = new double[3]{};
				(*GFJ2)[i][j][k] = new double[3]{};
			}
		}
	}

}

void mem_dealloc_fluxJacobian(double***** FFJ1, double***** FFJ2, double***** GFJ1, double***** GFJ2) {
	for (int i = 0; i < IMAX + 2; i++) {
		for (int j = 0; j < JMAX + 2; j++) {
			for (int k = 0; k < 3; k++) {

				delete[](*FFJ1)[i][j][k];
				delete[](*FFJ2)[i][j][k];
				delete[](*GFJ1)[i][j][k];
				delete[](*GFJ2)[i][j][k];
			

			}
			delete[](*FFJ1)[i][j];
			delete[](*FFJ2)[i][j];
			delete[](*GFJ1)[i][j];
			delete[](*GFJ2)[i][j];
		
		}
		delete[](*FFJ1)[i];
		delete[](*FFJ2)[i];
		delete[](*GFJ1)[i];
		delete[](*GFJ2)[i];
		

	}
	delete[](*FFJ1);
	delete[](*FFJ2);
	delete[](*GFJ1);
	delete[](*GFJ2);
}


void memory_allocation() {
	/* -----3D arrays ---- */
	{
		/* ------ IMAX +2 * JMAX + 2------- */
		Soln = { new double** [IMAX + 2] {} };
		dSoln = { new double** [IMAX + 2] {} };
		FI = { new double** [IMAX + 2] {} };

		Soln2 = { new double** [IMAX + 2] {} };


		for (int i = 0; i < IMAX + 2; i++) {
			Soln[i] = new double* [JMAX + 2];
			dSoln[i] = new double* [JMAX + 2];
			FI[i] = new double* [JMAX + 2];

			Soln2[i] = new double* [JMAX + 2];


			for (int j = 0; j < JMAX + 2; j++) {
				Soln[i][j] = new double[3]{};
				dSoln[i][j] = new double[3]{};
				FI[i][j] = new double[3]{};

				Soln2[i][j] = new double[3]{};

			}
		}

	}
	{		/* --------------4D arrays----------- */
		Bx = new double*** [IMAX + 2]{};
		Cx = new double*** [IMAX + 2]{};
		Ax = new double*** [IMAX + 2]{};
		By = new double*** [IMAX + 2]{};
		Cy = new double*** [IMAX + 2]{};
		Ay = new double*** [IMAX + 2]{};

		for (int i = 0; i < IMAX + 2; i++) {
			Bx[i] = new double** [JMAX + 2]{};
			Cx[i] = new double** [JMAX + 2]{};
			Ax[i] = new double** [JMAX + 2]{};
			By[i] = new double** [JMAX + 2]{};
			Cy[i] = new double** [JMAX + 2]{};
			Ay[i] = new double** [JMAX + 2]{};

			for (int j = 0; j < JMAX + 2; j++) {
				Bx[i][j] = new double* [3]{};
				Cx[i][j] = new double* [3]{};
				Ax[i][j] = new double* [3]{};
				By[i][j] = new double* [3]{};
				Cy[i][j] = new double* [3]{};
				Ay[i][j] = new double* [3]{};

				for (int k = 0; k < 3; k++) {
					Bx[i][j][k] = new double[3]{};
					Cx[i][j][k] = new double[3]{};
					Ax[i][j][k] = new double[3]{};
					By[i][j][k] = new double[3]{};
					Cy[i][j][k] = new double[3]{};
					Ay[i][j][k] = new double[3]{};
				}
			}
		}

	}
}


void memory_deallocate() {

	/* ------------------3D arrays -----------*/
	{
		/* ------ IMAX + 2 * JMAX + 2------- */
		for (int i = 0; i < IMAX + 2; i++) {
			for (int j = 0; j < JMAX + 2; j++) {
				delete[] Soln[i][j];
				delete[] dSoln[i][j];
				delete[] FI[i][j];

				delete[] Soln2[i][j];

			}
			delete[] Soln[i];
			delete[] dSoln[i];
			delete[] FI[i];

			delete[] Soln2[i];

		}
		delete[] Soln;
		delete[] dSoln;
		delete[] FI;

		delete[] Soln2;



	}

	{/*  -----------------4D arrays---------  */
		for (int i = 0; i < IMAX + 2; i++) {
			for (int j = 0; j < JMAX + 2; j++) {
				for (int k = 0; k < 3; k++) {

					delete[] Ax[i][j][k];
					delete[] Bx[i][j][k];
					delete[] Cx[i][j][k];
					delete[] Ay[i][j][k];
					delete[] By[i][j][k];
					delete[] Cy[i][j][k];

				}
				delete[] Ax[i][j];
				delete[] Bx[i][j];
				delete[] Cx[i][j];
				delete[] Ay[i][j];
				delete[] By[i][j];
				delete[] Cy[i][j];
			}
			delete[] Ax[i];
			delete[] Bx[i];
			delete[] Cx[i];
			delete[] Ay[i];
			delete[] By[i];
			delete[] Cy[i];

		}
		delete[] Ax;
		delete[] Bx;
		delete[] Cx;
		delete[] Ay;
		delete[] By;
		delete[] Cy;
	}

}





/* IO */
void write_col(FILE* file, const char* filename, double** array2d, int imax, int jmax) {


	file = fopen(filename, "w");
	fprintf(file, "%2s\t", "j");
	for (int j = 0; j < jmax; j++)
		fprintf(file, "%12d\t", j);
	fprintf(file, "\n");
	for (int i = 0; i < imax; i++) {
		fprintf(file, "i=%2d\t", i);
		for (int j = 0; j < jmax; j++) {
			fprintf(file, "%12G\t", array2d[i][j]);
		}
		fprintf(file, "\n");
	}
	fclose(file);

}

void write_row(FILE* file, const char* filename, double*** array2d, int imax, int jmax, int component)
{


	file = fopen(filename, "w");
	fprintf(file, "%2s\t", "i");
	for (int i = 0; i < imax; i++)
		fprintf(file, "%12d\t", i);
	fprintf(file, "\n");
	for (int j = 0; j < jmax; j++) {
		fprintf(file, "j=%2d\t", j);
		for (int i = 0; i < imax; i++) {
			fprintf(file, "%12G\t", array2d[i][j][component]);
		}
		fprintf(file, "\n");
	}
	fclose(file);

}

void write_row2(FILE* file, const char* filename, double** array2d, double** array2d2, int imax, int jmax)
{


	file = fopen(filename, "w");
	fprintf(file, "%2s\t", "i");
	for (int i = 0; i < imax; i++)
		fprintf(file, "%12d\t", i);
	fprintf(file, "\n");
	for (int j = 0; j < jmax; j++) {
		fprintf(file, "j=%2d\t", j);
		for (int i = 0; i < imax; i++) {
			fprintf(file, "%12G\t", -array2d[i][j] - array2d2[i][j]);
		}
		fprintf(file, "\n");
	}
	fclose(file);

}

void write_row_error(FILE* file, const char* filename, double** array2d, double** array2d2, double** exact, int imax, int jmax)
{


	file = fopen(filename, "w");
	fprintf(file, "%2s\t", "i");
	for (int i = 0; i < imax; i++)
		fprintf(file, "%12d\t", i);
	fprintf(file, "\n");
	for (int j = 0; j < jmax; j++) {
		fprintf(file, "j=%2d\t", j);
		for (int i = 0; i < imax; i++) {
			double exact_FI = exact[i][j];
			double computed = array2d[i][j] + array2d2[i][j];
			double error = exact_FI - computed;
			fprintf(file, "%12G\t", computed);
		}
		fprintf(file, "\n");
	}
	fclose(file);

}

void write_row_1d(FILE* file, const char* filename, double array1d[IMAX + 2][3], int l, int component)
{
	file = fopen(filename, "w");
	fprintf(file, "%2s\n", "i");
	for (int i = 0; i < l; i++)
	{
		fprintf(file, "%2d %12G\n", i, array1d[i][component]);
	}
	fclose(file);
}

void write_sym_plot_u(FILE* file, const char* filename, double*** array1d, int l) {
	file = fopen(filename, "w");

	/* calculating i index and X location of left and right of the center of squre channel */
	int mid_left = IMAX / 2.0 + 0.5;
	double X_left = (mid_left - 0.5) * dX;
	int mid_right = IMAX / 2.0 + 1.5;
	double X_right = (mid_right - 0.5) * dX;

	double X_mid = (X_left + X_right) / 2.0;

	fprintf(file, "%4s %12s  at X = %4f\n", "Y", "u_mid", X_mid);
	/* setting the speed at bottom wall */
	fprintf(file, "%4f %12G\n", 0.0, 0.0);
	for (int j = 1; j < l - 1; j++)
	{
		double Y = (j - 0.5) * dY;
		double u_mid = (array1d[mid_right][j][1] + array1d[mid_left][j][1]) / 2.0;
		fprintf(file, "%4f %12G\n", Y, u_mid);
	}

	/* avergaing top moving wall speed, using the solution in the approximity */
	double u_right = (array1d[mid_right][JMAX+1][1] + array1d[mid_right][JMAX][1]) / 2.0;
	double u_left = (array1d[mid_left][JMAX + 1][1] + array1d[mid_left][JMAX][1]) / 2.0;
	double u_mid = (u_right + u_left) / 2.0;

	fprintf(file, "%4f %12G\n", 1.0, u_mid);

	fclose(file);
}

void write_tecplot(FILE* file, const char* filename, double*** array)
{
	file = fopen(filename, "w");

	fprintf(file, "variables=X, Y, P, u, v, vel, vorticity\n");
	fprintf(file, "ZONE\n");
	fprintf(file, "i=%2d, j=%2d, F=POINT\n", IMAX+2, JMAX+2);

	
	for (int j = 0; j < JMAX + 2; j++) {
		double Y{};
		if (j == 0)
			Y = 0.0;
		else if (j == JMAX + 1)
			Y = Height;
		else
			Y = (j - 0.5) * dY;

		for (int i = 0; i < IMAX + 2; i++) {
			double X{};
			if (i == 0)
				X = 0.0;
			else if (i == IMAX + 1)
				X = Width;
			else
				X = (i - 0.5) * dX;

			if (j == 0) {
				double P = (array[i][j][0] + array[i][j + 1][0]) / 2.0;
				double u = (array[i][j][1] + array[i][j + 1][1]) / 2.0;
				double v = (array[i][j][2] + array[i][j + 1][2]) / 2.0;
				double vel = sqrt(u * u + v * v);
				double vort = 0;

				fprintf(file, "%5f %5f %12e %12e %12e %12e %12e\n", X, Y, P, u, v, vel, vort);
			}
			else if (j == JMAX + 1) {
				if (i == 0 or i == IMAX + 1) {
					double P = (array[i][j][0] + array[i][j - 1][0]) / 2.0;
					double u = -(array[i][j][1] + array[i][j - 1][1]) / 2.0;
					double v = (array[i][j][2] + array[i][j - 1][2]) / 2.0;
					double vel = sqrt(u * u + v * v);
					double vort = 0;

					fprintf(file, "%5f %5f %12e %12e %12e %12e %12e\n", X, Y, P, u, v, vel, vort);
				}
				else {
					double P = (array[i][j][0] + array[i][j - 1][0]) / 2.0;
					double u = (array[i][j][1] + array[i][j - 1][1]) / 2.0;
					double v = (array[i][j][2] + array[i][j - 1][2]) / 2.0;
					double vel = sqrt(u * u + v * v);
					double vort = 0;

					fprintf(file, "%5f %5f %12e %12e %12e %12e %12e\n", X, Y, P, u, v, vel, vort);
				}
			}
			else {
				if (i == 0) {
					double P = (array[i][j][0] + array[i + 1][j][0]) / 2.0;
					double u = (array[i][j][1] + array[i + 1][j][1]) / 2.0;
					double v = (array[i][j][2] + array[i + 1][j][2]) / 2.0;
					double vel = sqrt(u * u + v * v);
					double vort = 0;

					fprintf(file, "%5f %5f %12e %12e %12e %12e %12e\n", X, Y, P, u, v, vel, vort);
				}
				else if (i == IMAX + 1) {
					double P = (array[i][j][0] + array[i - 1][j][0]) / 2.0;
					double u = (array[i][j][1] + array[i - 1][j][1]) / 2.0;
					double v = (array[i][j][2] + array[i - 1][j][2]) / 2.0;
					double vel = sqrt(u * u + v * v);
					double vort = 0;
					fprintf(file, "%5f %5f %12e %12e %12e %12e %12e\n", X, Y, P, u, v, vel, vort);
				}
				else {
					double P = array[i][j][0];
					double u = array[i][j][1];
					double v = array[i][j][2];
					double vel = sqrt(u * u + v * v);
					double vort = fabs((Soln[i][j + 1][1] - Soln[i][j - 1][1]) - (Soln[i + 1][j][2] - Soln[i - 1][j][2]));
					fprintf(file, "%5f %5f %12e %12e %12e %12e %12e\n", X, Y, P, u, v, vel, vort);
				}
			}
		}
	}
	
	fclose(file);
}



//void write_tecplot_diff(FILE* file, const char* filename, double*** array, double*** array2)
//{
//	file = fopen(filename, "w");
//
//	fprintf(file, "variables=X, Y, dP, du, dv, dvel\n");
//	fprintf(file, "ZONE\n");
//	fprintf(file, "i=%2d, j=%2d, F=POINT\n", IMAX + 2, JMAX + 2);
//
//
//	for (int j = 0; j < JMAX + 2; j++) {
//		
//		for (int i = 0; i < IMAX + 2; i++) {
//			double X{};
//			if (i == 0)
//				X = 0.0;
//			else if (i == IMAX + 1)
//				X = Width;
//			else
//				X = (i - 0.5) * dX;
//
//			if (j == 0) {
//				double P = (array[i][j][0] + array[i][j + 1][0]) / 2.0;
//				double u = (array[i][j][1] + array[i][j + 1][1]) / 2.0;
//				double v = (array[i][j][2] + array[i][j + 1][2]) / 2.0;
//				double vel = sqrt(u * u + v * v);
//
//				double P2 = (array2[i][j][0] + array2[i][j + 1][0]) / 2.0;
//				double u2 = (array2[i][j][1] + array2[i][j + 1][1]) / 2.0;
//				double v2 = (array2[i][j][2] + array2[i][j + 1][2]) / 2.0;
//				double vel2 = sqrt(u2 * u2 + v2 * v2);
//				fprintf(file, "%5f %5f %12f %12f %12f %12f\n", X, Y, P+P2, u+u2, v+v2, vel+vel2);
//			}
//			else if (j == JMAX + 1) {
//				if (i == 0 or i == IMAX + 1) {
//					double P = (array[i][j][0] + array[i][j - 1][0]) / 2.0;
//					double u = -(array[i][j][1] + array[i][j - 1][1]) / 2.0;
//					double v = (array[i][j][2] + array[i][j - 1][2]) / 2.0;
//					double vel = sqrt(u * u + v * v);
//
//					double P2 = (array2[i][j][0] + array2[i][j - 1][0]) / 2.0;
//					double u2 = -(array2[i][j][1] + array2[i][j - 1][1]) / 2.0;
//					double v2 = (array2[i][j][2] + array2[i][j - 1][2]) / 2.0;
//					double vel2 = sqrt(u2 * u2 + v2 * v2);
//					fprintf(file, "%5f %5f %12f %12f %12f %12f\n", X, Y, P + P2, u + u2, v + v2, vel + vel2);
//				}
//				else {
//					double P = (array[i][j][0] + array[i][j - 1][0]) / 2.0;
//					double u = (array[i][j][1] + array[i][j - 1][1]) / 2.0;
//					double v = (array[i][j][2] + array[i][j - 1][2]) / 2.0;
//					double vel = sqrt(u * u + v * v);
//
//					double P2 = (array2[i][j][0] + array2[i][j - 1][0]) / 2.0;
//					double u2 = (array2[i][j][1] + array2[i][j - 1][1]) / 2.0;
//					double v2 = (array2[i][j][2] + array2[i][j - 1][2]) / 2.0;
//					double vel2 = sqrt(u2 * u2 + v2 * v2);
//					fprintf(file, "%5f %5f %12f %12f %12f %12f\n", X, Y, P + P2, u + u2, v + v2, vel + vel2);
//				}
//			}
//			else {
//				if (i == 0) {
//					double P = (array[i][j][0] + array[i + 1][j][0]) / 2.0;
//					double u = (array[i][j][1] + array[i + 1][j][1]) / 2.0;
//					double v = (array[i][j][2] + array[i + 1][j][2]) / 2.0;
//					double vel = sqrt(u * u + v * v);
//
//					double P2 = (array2[i][j][0] + array2[i + 1][j][0]) / 2.0;
//					double u2 = (array2[i][j][1] + array2[i + 1][j][1]) / 2.0;
//					double v2 = (array2[i][j][2] + array2[i + 1][j][2]) / 2.0;
//					double vel2 = sqrt(u2 * u2 + v2 * v2);
//					fprintf(file, "%5f %5f %12f %12f %12f %12f\n", X, Y, P + P2, u + u2, v + v2, vel + vel2);
//				}
//				else if (i == IMAX + 1) {
//					double P = (array[i][j][0] + array[i - 1][j][0]) / 2.0;
//					double u = (array[i][j][1] + array[i - 1][j][1]) / 2.0;
//					double v = (array[i][j][2] + array[i - 1][j][2]) / 2.0;
//					double vel = sqrt(u * u + v * v);
//
//					double P2 = (array2[i][j][0] + array2[i - 1][j][0]) / 2.0;
//					double u2 = (array2[i][j][1] + array2[i - 1][j][1]) / 2.0;
//					double v2 = (array2[i][j][2] + array2[i - 1][j][2]) / 2.0;
//					double vel2 = sqrt(u2 * u2 + v2 * v2);
//					fprintf(file, "%5f %5f %12f %12f %12f %12f\n", X, Y, P + P2, u + u2, v + v2, vel + vel2);
//				}
//				else {
//					
//				}
//			}
//		}
//	}
//
//	fclose(file);
//}


void write_tecplot_diff(FILE* file, const char* filename, double*** array, double*** array2)
{
	file = fopen(filename, "w");

	fprintf(file, "variables=X, Y, dP, du, dv, dvel\n");
	fprintf(file, "ZONE\n");
	fprintf(file, "i=%2d, j=%2d, F=POINT\n", IMAX , JMAX );


	for (int j = 1; j < JMAX + 1; j++) {
		double Y = (j - 0.5) * dY;
		for (int i = 1; i < IMAX + 1; i++) {
			double X = (i - 0.5) * dX;

			double P = array[i][j][0];
			double u = array[i][j][1];
			double v = array[i][j][2];
			double vel = sqrt(u * u + v * v);

			double P2 = array2[IMAX+1-i][j][0];
			double u2 = array2[IMAX+1-i][j][1];
			double v2 = array2[IMAX+1-i][j][2];
			double vel2 = sqrt(u2 * u2 + v2 * v2);
			fprintf(file, "%5f %5f %16e %16.16e %16e %16e\n", X, Y, P + P2, u + u2, v + v2, vel + vel2);
		}
	}

	fclose(file);
}

//void mem_alloc_AF(double***** LHS, double*** RHS) {
//	(*LHS) = new double*** [JMAX + 2]{};
//	for (int i = 0; i < JMAX + 2; i++) {
//		(*LHS)[i] = new double** [3]{};
//		for (int j = 0; j < 3; j++) {
//			(*LHS)[i][j] = new double* [3]{};
//			for (int k = 0; k < 3; k++) {
//				(*LHS)[i][j][k] = new double[3]{};
//			}
//		}
//	}
//
//	(*RHS) =  new double* [JMAX + 2] {} ;
//	for (int i = 0; i < JMAX + 2; i++) {
//		(*RHS)[i] = new double [3];
//	}
//}
//
//void mem_dealloc_AF(double***** LHS, double*** RHS) {
//	for (int i = 0; i < JMAX + 2; i++) {
//		for (int j = 0; j < 3; j++) {
//			for (int k = 0; k < 3; k++) {
//				delete[](*LHS)[i][j][k];
//			}
//			delete[](*LHS)[i][j];
//		}
//		delete[](*LHS)[i];
//	}
//	delete[](*LHS);
//
//	for (int i = 0; i < JMAX + 2; i++) {
//			delete[](*RHS)[i];
//	}
//	delete[] *RHS;
//}