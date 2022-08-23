/*
 * DMD.h
 *
 *  Created on: Jul 30, 2022
 *      Author: mirshahi
 */

#ifndef APPS_CONVRATESPEEDUP_DMD_H_
#define APPS_CONVRATESPEEDUP_DMD_H_

#include <stdio.h>
#include <memory>
#include <chrono>
#include <math.h>
#include <vector>
#include <complex.h>
#include <cassert>

#include <petscksp.h>
#include <slepcsvd.h>
#include <slepceps.h>
#include <cblas.h>
#include <lapacke.h>

//#define DEBUG_DMD
//#define DEBUG_DMD_EPS
#define PRINT_EIGENVALUES
#define DMD_SIGMARATIO

class DMD {
public:
//	DMD(const UnstructuredMesh *pUnstructuredMesh, Physics *const phys,
//			PetscReal DT, Mat *Data);

	DMD(const Mat *Data, PetscInt iModes, PetscReal DT);
	virtual ~DMD();
	PetscErrorCode prepareData();

	PetscErrorCode solveSVD();
	PetscErrorCode regression();
	PetscErrorCode calcDMDmodes();
	PetscErrorCode computeUpdate(PetscInt iMode);
	PetscErrorCode computeMatTransUpdate();

	PetscErrorCode applyDMD();
	PetscErrorCode applyDMDMatTrans();


	Vec vgetUpdate() {
		return update;
	}

	const Mat& mGetDMDModes() const {
		return Phi;
	}

	int iGetSVDRank() const {
		return svdRank;
	}

	PetscErrorCode lapackMatInv(Mat &A);

	/* -----  Print functions  ------ */
	PetscErrorCode printMatMATLAB(std::string sFilename,
			std::string sMatrixName, Mat A) const;
	PetscErrorCode printVecMATLAB(std::string sFileName,
			std::string sVectorName, Vec V) const;
	PetscErrorCode printVecPYTHON(std::string sFileName,
			std::string sVectorName, Vec V) const;
	PetscErrorCode printMatPYTHON(std::string sFilename,
			std::string sMatrixName, Mat A) const;


private:

	const int reconOrder{};


	PetscInt *row_index{};
	PetscInt svdRank{}; //number of columns in SVD modes, also the truncation of SVD
	PetscInt iNumModes{};
	PetscReal dt;

	Mat X1 = PETSC_NULL, X2 = PETSC_NULL;
	Mat Ur = PETSC_NULL, Sr = PETSC_NULL, Vr = PETSC_NULL;
	Mat Sr_inv = NULL, W = NULL;
	Mat Atilde = PETSC_NULL, Phi = PETSC_NULL, time_dynamics = PETSC_NULL;

	Vec update = NULL;

	std::vector<std::complex<double>> eigs;

	struct _DATA{
		const Mat  *mat{};
		PetscInt num_cols, num_rows;
	};

	_DATA X;

};


#endif /* APPS_CONVRATESPEEDUP_DMD_H_ */
