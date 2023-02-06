/*
 * DMD.h
 *
 *  Created on: Jul 30, 2022
 *      Author: mirshahi
 */

#ifndef APPS_CONVRATESPEEDUP_DMD_H_
#define APPS_CONVRATESPEEDUP_DMD_H_

#include <iostream>
#include <stdio.h>
#include <memory>
#include <chrono>
#include <math.h>
#include <vector>
#include <complex.h>
#include <cassert>

#include <sys/stat.h>
#include <fstream>

#include <petscksp.h>
#include <slepcsvd.h>
#include <slepceps.h>
#include <cblas.h>
#include <lapacke.h>

#include "petscmat.h"
//#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

//#define DEBUG_DMD
//#define DEBUG_DMD_EPS
//#define TEST_EIGEN_SYSTEMS
//#define DMD_CHECK_EIGS
//#define DMD_SIGMARATIO
//#define CALC_CONDITION_NUMBER_OF_UPDATE
//#define TIMING
//#define SNAPS_SUB_MEAN
//#define SNAPS_VEC_NORMALIZE //normalize snapshot vectors
//#define WRITE_SNAP_MAT

#define ML



#define CORETHRESH 0.0

# define M_PIl          3.141592653589793238462643383279502884L /* pi */

using ComplexNum = std::complex<double>;
using ComplexSTLVec = std::vector<ComplexNum>;

class DMD {
private:
	PetscInt *row_index{};
	PetscInt svdRank{}; //number of columns in SVD modes, also the truncation of SVD
//	PetscInt iNumModes{}; // Outdated parameter - First need to fix calcDMDmodes() member function
	PetscReal dt;
	PetscReal iterNorm{-1}; // Norm of the current solution
	PetscInt iOsclPeriod = 0; // Oscillation period of the dominant Mode
	PetscReal dFroNorm{}, dInfNorm{};
	FILE* fLog;
	std::ofstream fML;


	PetscBool flg_autoRankDMD = PETSC_FALSE; // automate dmd matrix manipulation

	Mat X1 = PETSC_NULL, X2 = PETSC_NULL;
	Mat Atilde = PETSC_NULL, time_dynamics = PETSC_NULL;
//	Eigen::MatrixXcd eigenPhi; // DMD modes - Eigen3 Matrix (em)
	Eigen::MatrixXcd epsPhi;

	Vec update = NULL;


	struct _svd{
		Mat Ur = PETSC_NULL, Sr = PETSC_NULL, Vr = PETSC_NULL;
		Mat Sr_inv = NULL, W = NULL;
		Eigen::VectorXcd omega_sorted; // Dominant eigenvalues of the solution updates
		Eigen::MatrixXcd eigVecs_small; //Eigenvectors in the small space of DMD
		Eigen::VectorXcd eigs; //Dominant amplification factors
	};

	struct _DATA{
		const Mat  *mat{};
		PetscInt num_cols, num_rows;
	};

	_svd lrSVD, fullSVD;
	_DATA X;

	const std::string DEB_DIR{"debug_tools"};
	const std::string DEB_MAT_DIR{"debug_tools/mats/"};
	const std::string DEB_TOOL_DIR{"debug_tools/tools/"};

	/* Count the number of times this class was called */
	static int isDMD_execs;

public:

	DMD(const Mat *Data, PetscReal DT, PetscReal dNorm);
	virtual ~DMD();
	PetscErrorCode prepareData();

	PetscErrorCode regression(bool dummyDMD = false);
	PetscErrorCode calcDMDmodes(); // does not include complex numbers - should be fixed!!
	PetscErrorCode computeMatUpdate();

	PetscErrorCode applyDMDMatTrans();
	PetscErrorCode DummyDMD();

//	PetscErrorCode computeUpdate(PetscInt iMode);
//	PetscErrorCode applyDMD(); Outdated parameter - First need to fix calcDMDmodes() member function

	Vec vgetUpdate() {
		return update;
	}

	Eigen::MatrixXcd mGetDMDModes() const {
		return epsPhi;
	}

	PetscInt iGetSVDRank() const {
		return svdRank;
	}

	PetscInt iGetDominantPeriod() const {
		return iOsclPeriod;
	}


	PetscErrorCode solveSVD(SVD& svd, Mat& mMatrix);
	PetscErrorCode computeSVDRank(SVD &svd);
	PetscErrorCode calcLowRankSVDApprox(SVD &svd, PetscInt rank, _svd &LowSVD,
			std::string sFileName, bool squreMat = false);
	PetscErrorCode calcBestFitlrSVD(_svd &LowSVD, Mat &mBestFit);
	PetscErrorCode calcEigenvalues(_svd &LowSVD, Mat &matrix,
			std::string sFileName, bool calcEigenvectors = false);
	PetscErrorCode calcDominantModePeriod(Mat& matrix);

	PetscErrorCode lapackMatInv(Mat &A);
	PetscErrorCode calcUpdateNorm(const _svd &LowSVD,
			const Mat &mAtilde, const Mat &mX1, const Mat &mX2);
	PetscErrorCode dotwDMDmodes(const Vec& pVec, int numMode, bool bEPS);

	/* -----  Print functions  ------ */
	PetscErrorCode printMatMATLAB(std::string sFilename,
			std::string sMatrixName, Mat A) const;
	PetscErrorCode printVecMATLAB(std::string sFileName,
			std::string sVectorName, Vec V) const;
	PetscErrorCode printVecPYTHON(std::string sFileName,
			std::string sVectorName, Vec V) const;
	PetscErrorCode printMatPYTHON(std::string sFilename,
			std::string sMatrixName, Mat A) const;
	PetscErrorCode testSVD(SVD& svd);
	PetscErrorCode testEPS();

	void recordTime(std::chrono::steady_clock::time_point start,
			std::string sMessage);

	bool IsPathExist(const std::string &s)
	{
	  struct stat buffer;
	  return (stat (s.c_str(), &buffer) == 0);
	}

};




#endif /* APPS_CONVRATESPEEDUP_DMD_H_ */
