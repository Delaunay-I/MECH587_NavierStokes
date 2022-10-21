/*
 * DMD.cpp
 *
 *  Created on: Jul 30, 2022
 *      Author: mirshahi
 */

#include "DMD.h"

int DMD::isDMD_execs = 0;

DMD::DMD(const Mat *svdMat, PetscReal DT) :
	 dt(DT) {

	/* Opening a log file to write some results, to access after code execution */
	fLog = fopen("Log.dat", "a");

	/* Set the snapshots matrix and get the sizes of its rows and columns */
	X.mat = svdMat;
	/* Data is row-major, computations are based on column-major */
	MatGetSize(*X.mat, &X.num_cols, &X.num_rows);
	PetscMalloc1(X.num_rows, &row_index);

	for (int i = 0; i < X.num_rows; i++) {
		row_index[i] = i;
	}

	/* Get the number of modes to eliminate at each call to the DMD class */
	PetscBool flg;
	PetscInt iDMD, *ipDMDRanks{};

	PetscOptionsHasName(NULL, PETSC_NULL, "-DMD_autoSetMatrix", &flg_autoRankDMD);

	if (!flg_autoRankDMD){
		PetscOptionsGetInt(NULL, NULL, "-DMD_nits", &iDMD, &flg);
		PetscMalloc1(iDMD, &ipDMDRanks);
		PetscOptionsGetIntArray(PETSC_NULL, PETSC_NULL, "-DMD_ranks", ipDMDRanks, &iDMD, &flg);
		if (!flg) {
			PetscErrorPrintf("Missing -dmd_ranks flag!\n");
			exit(2);
		} else if (svdRank > X.num_cols - 1) {
			PetscErrorPrintf(
					"\nSVD rank is greater than the number of (snapshots columns - 1).\n"
							"Decrease the SVD rank, or increase the number of snapshots columns.\n");
			exit(2);
		}
		svdRank = ipDMDRanks[isDMD_execs];
	}


	/* creating a few directories */
	if (!IsPathExist(DEB_DIR)) {
		const int dir_err = mkdir(DEB_DIR.c_str(),
				S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		if (-1 == dir_err) {
			printf("Error creating directory %s\n", DEB_MAT_DIR.c_str());
			exit(1);
		}
	}

	if (!IsPathExist(DEB_MAT_DIR)) {
		const int dir_err = mkdir(DEB_MAT_DIR.c_str(),
				S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		if (-1 == dir_err) {
			printf("Error creating directory %s\n", DEB_MAT_DIR.c_str());
			exit(1);
		}
	}

	if (!IsPathExist(DEB_TOOL_DIR)) {
		const int dir_err = mkdir(DEB_TOOL_DIR.c_str(),
				S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		if (-1 == dir_err) {
			printf("Error creating directory %s\n", DEB_TOOL_DIR.c_str());
			exit(1);
		}
	}

}



DMD::~DMD() {
	/* Writing information to our log file */
	fprintf(fLog, "SVDRank: %i\t num Snapshots: %i-1\n", svdRank, X.num_cols);

	fprintf(fLog, "\n");
	fclose(fLog);
	PetscFree(row_index);
	MatDestroy(&X1);
	MatDestroy(&X2);
	MatDestroy(&X2_tilde);
	MatDestroy(&lrSVD.Ur);
	MatDestroy(&lrSVD.Sr);
	MatDestroy(&lrSVD.Vr);
	MatDestroy(&lrSVD.Sr_inv);
	MatDestroy(&lrSVD.W);
	MatDestroy(&Atilde);
	MatDestroy(&Phi);
	MatDestroy(&time_dynamics);
	VecDestroy(&update);

}


/*
 * Splitting data into 2 matrices
 * Hence the definition of the rows and columns are opposite
 */
PetscErrorCode DMD::prepareData(){
	PetscErrorCode ierr;
	PetscInt *X1row_index, *X2row_index;
	PetscScalar *dTmpArr;
	PetscInt rows{X.num_cols}, cols{X.num_rows};

	ierr = PetscMalloc1(rows - 1, &X1row_index); CHKERRQ(ierr);
	ierr = PetscMalloc1(rows - 1, &X2row_index); CHKERRQ(ierr);
	ierr = PetscMalloc1(cols*(rows - 1), &dTmpArr); CHKERRQ(ierr);


	for (int i = 0; i < rows - 1; i++) {
		X1row_index[i] = i; //X1col_index is used for both X1 and X2 for setting values
		X2row_index[i] = i + 1;
	}

	ierr = MatDestroy(&X1);	CHKERRQ(ierr);
	ierr = MatDestroy(&X2);	CHKERRQ(ierr);

	ierr = MatCreate(MPI_COMM_WORLD, &X1);	CHKERRQ(ierr);
	ierr = MatSetSizes(X1, PETSC_DECIDE, PETSC_DECIDE, rows - 1,
			cols);	CHKERRQ(ierr);
	ierr = MatSetType(X1, MATAIJ); CHKERRQ(ierr);
	ierr = MatSetUp(X1);	CHKERRQ(ierr);

	ierr = MatCreate(MPI_COMM_WORLD, &X2);	CHKERRQ(ierr);
	ierr = MatSetSizes(X2, PETSC_DECIDE, PETSC_DECIDE, rows - 1,
			cols);	CHKERRQ(ierr);
	ierr = MatSetType(X2, MATAIJ); CHKERRQ(ierr);
	ierr = MatSetUp(X2);	CHKERRQ(ierr);

	ierr = MatGetValues(*X.mat, rows - 1, X1row_index, cols, row_index, dTmpArr);
	ierr = MatSetValues(X1, rows - 1, X1row_index, cols, row_index, dTmpArr,
			INSERT_VALUES);
	CHKERRQ(ierr);

	ierr = MatAssemblyBegin(X1, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);
	ierr = MatAssemblyEnd(X1, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);

	ierr = MatGetValues(*X.mat, rows - 1, X2row_index, cols, row_index, dTmpArr);
	ierr = MatSetValues(X2, rows - 1, X1row_index, cols, row_index, dTmpArr,
			INSERT_VALUES);
	CHKERRQ(ierr);

	ierr = MatAssemblyBegin(X2, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);
	ierr = MatAssemblyEnd(X2, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);

	/* Make the matrices tall-skinny for compatibility with other functions */
	ierr = MatTranspose(X1, MAT_INPLACE_MATRIX, &X1); CHKERRQ(ierr);
	ierr = MatTranspose(X2, MAT_INPLACE_MATRIX, &X2); CHKERRQ(ierr);


	ierr = PetscFree(X1row_index); CHKERRQ(ierr);
	ierr = PetscFree(X2row_index); CHKERRQ(ierr);
	ierr = PetscFree(dTmpArr); CHKERRQ(ierr);

	return ierr;
}

PetscErrorCode DMD::regression(bool dummyDMD) {
	PetscErrorCode ierr;
	SVD svd;

	auto start = std::chrono::steady_clock::now();

	ierr = solveSVD(svd, X1); CHKERRQ(ierr);
	std::string sMessage = "DMD - SVD time";
	recordTime(start, sMessage);

	// Compute the proper rank if requested
	if (flg_autoRankDMD) {
		ierr = computeSVDRank(svd); CHKERRQ(ierr);
	}

	printf("DMD was called %i times, current SVD approximation rank: %i\n",
			isDMD_execs + 1, svdRank);

	/*
	 * Getting Singular values and singular vectors
	 * and building low rank truncation matrices
	 */
	ierr = PetscPrintf(PETSC_COMM_WORLD,
					"computing the %i-rank-SVD-approximation of the snapshots matrix.\n", svdRank); CHKERRQ(ierr);
		ierr = calcLowRankSVDApprox(svd, svdRank, lrSVD, "SnapshotsMat-LowRank");
		CHKERRQ(ierr);

#ifdef DMD_CHECK_EIGS
	if (!dummyDMD) {
		Mat fullAtilde;
		ierr = calcLowRankSVDApprox(svd, X.num_cols - 1, fullSVD, "SnapshotsMat-Full"); CHKERRQ(ierr);
		/* Finds the FULL Atilde using all the columns (all the modes) */
		ierr = calcBestFitlrSVD(fullSVD, fullAtilde); CHKERRQ(ierr);
		/* Calculates the eigenvalues of the FULL Atilde */
		ierr = calcEigenvalues(fullSVD, fullAtilde, "fullAtilde_eigenvalues"); CHKERRQ(ierr);
		MatDestroy(&fullAtilde);
	}
#endif

	ierr = calcBestFitlrSVD(lrSVD, Atilde); CHKERRQ(ierr);
	/* ------------ Eigen analysis of Atilde -------------*/
	if (!dummyDMD) {
		ierr = calcEigenvalues(lrSVD, Atilde, "LowRank-Eigenvalues", true); CHKERRQ(ierr);
	} else {
		ierr = calcDominantModePeriod(Atilde); CHKERRQ(ierr);
	}

	return ierr;
}



/*
 * This function does not consider complex numbers - SHOULD BE FIXED!!!
 */

//PetscErrorCode DMD::calcDMDmodes(){
//	PetscErrorCode ierr;
//	Mat X2_Vr;
//
//	// Calculating Spatial modes
//		ierr = MatMatMult(X2, lrSVD.Vr, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &X2_Vr); CHKERRQ(ierr);
//		ierr = MatMatMatMult(X2_Vr, lrSVD.Sr_inv, lrSVD.W, MAT_INITIAL_MATRIX, PETSC_DEFAULT,
//						&Phi); CHKERRQ(ierr);
//
//		ComplexSTLVec omega;
//		for (size_t i = 0; i < lrSVD.eigs.size(); i++){
//			ComplexNum complexEig = lrSVD.eigs[i];
//			ComplexNum tmp2 = log(complexEig);
//			assert(isfinite(std::real(tmp2)) && "Omega has infinite value!!\n\n");
//			omega.push_back(tmp2/dt);
//		}
//
//		std::ofstream fOMEGA_new;
//		fOMEGA_new.open("Omega_new.dat");
//		for(auto element: omega){
//			fOMEGA_new << element << std::endl;
//		}
//		fOMEGA_new.close();
//
//	Vec rhs, Soln; // rhs = x1, Soln = b
//	KSP ksp;
//	PC pc;
//	ierr = VecCreateSeq(PETSC_COMM_SELF, X.num_rows, &rhs); CHKERRQ(ierr);
//	ierr = VecCreateSeq(PETSC_COMM_SELF, svdRank, &Soln); CHKERRQ(ierr);
//	ierr = MatGetColumnVector(*X.mat, rhs, 0); CHKERRQ(ierr);
//
//#ifdef DEBUG_DMD
//	ierr = printVecMATLAB(DEB_MAT_DIR + "x1", "x1", rhs); CHKERRQ(ierr);
//	ierr = printVecMATLAB(DEB_MAT_DIR + "b", "b", Soln); CHKERRQ(ierr);
//
//#endif
//	ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
//
//	ierr = KSPSetOperators(ksp, Phi, Phi); CHKERRQ(ierr);
//	ierr = KSPSetType(ksp, KSPLSQR); CHKERRQ(ierr);
//	ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
//	ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);
//	ierr = KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
//
//	//ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
//
//	ierr = KSPSolve(ksp, rhs, Soln); CHKERRQ(ierr);
////	ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
//
//#ifdef DEBUG_DMD
//	/*
//	 * The Solution of the least squares (b) is related to the eigenvectors and
//	 * the fact that we are not including imaginery parts in the KSP solver
//	 * So the results are diffferent from Python and MATLAB
//	 */
//	ierr = printVecMATLAB(DEB_MAT_DIR + "b", "b", Soln); CHKERRQ(ierr);
//#endif
//
//	ierr = MatCreate(MPI_COMM_WORLD, &time_dynamics);	CHKERRQ(ierr);
//	ierr = MatSetSizes(time_dynamics, PETSC_DECIDE, PETSC_DECIDE, X.num_cols,
//			svdRank);	CHKERRQ(ierr);
//	ierr = MatSetType(time_dynamics, MATAIJ); CHKERRQ(ierr);
//	ierr = MatSetUp(time_dynamics);	CHKERRQ(ierr);
//
//	PetscReal t = 0;
//	for (int iter = 0; iter < X.num_cols; iter++) {
//		for (int mode = 0; mode < svdRank; mode++) {
//			PetscScalar bVal;
//			ierr = VecGetValues(Soln, 1, &mode, &bVal); CHKERRQ(ierr);
////			PetscScalar value = std::real(bVal*exp(omega[mode]*t));
////
////			ierr = MatSetValue(time_dynamics_old, iter, mode, value, INSERT_VALUES); CHKERRQ(ierr);
//
//			PetscScalar value = std::real(bVal*exp(omega[mode]*t));
//			ierr = MatSetValue(time_dynamics, iter, mode, value, INSERT_VALUES); CHKERRQ(ierr);
//
//		}
//		t += dt;
//	}
//	ierr = MatAssemblyBegin(time_dynamics_old, MAT_FINAL_ASSEMBLY);	CHKERRQ(ierr);
//	ierr = MatAssemblyEnd(time_dynamics_old, MAT_FINAL_ASSEMBLY);	CHKERRQ(ierr);
//	ierr = MatAssemblyBegin(time_dynamics, MAT_FINAL_ASSEMBLY);	CHKERRQ(ierr);
//	ierr = MatAssemblyEnd(time_dynamics, MAT_FINAL_ASSEMBLY);	CHKERRQ(ierr);
//
//#ifdef DEBUG_DMD
//	ierr = printMatPYTHON(DEB_MAT_DIR + "TimeDynamics_old", "TD", time_dynamics_old); CHKERRQ(ierr);
//	ierr = printMatPYTHON(DEB_MAT_DIR + "TimeDynamics", "TD", time_dynamics); CHKERRQ(ierr);
//#endif
//
//	ierr = MatDestroy(&X2_Vr); CHKERRQ(ierr);
//	ierr = VecDestroy(&rhs); CHKERRQ(ierr);
//	ierr = VecDestroy(&Soln); CHKERRQ(ierr);
//	ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
//	return ierr;
//}

PetscErrorCode DMD::computeMatTransUpdate() {
	PetscErrorCode ierr;
	Mat Ir{}, lhs{}, Gtilde{};

	ierr = calcUpdateNorm(lrSVD, Atilde, X1, X2); CHKERRQ(ierr);
	assert(X2_tilde != PETSC_NULL
			&& "X2_tilde isn't defined. Have you computed the norm? (X2_tilde is computed inside that function)");

	ierr = MatDuplicate(Atilde, MAT_COPY_VALUES, &Ir);
	CHKERRQ(ierr);
	ierr = MatDuplicate(Atilde, MAT_COPY_VALUES, &lhs);
	CHKERRQ(ierr);
	ierr = MatDuplicate(Atilde, MAT_COPY_VALUES, &Gtilde);
	CHKERRQ(ierr);
	ierr = MatZeroEntries(Ir);
	CHKERRQ(ierr);
	ierr = MatZeroEntries(Gtilde);
	CHKERRQ(ierr);

	for (int i = 0; i < svdRank; i++) {
		ierr = MatSetValue(Ir, i, i, 1.0, INSERT_VALUES);
		CHKERRQ(ierr);
	}
	ierr = MatAssemblyBegin(Ir, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);
	ierr = MatAssemblyEnd(Ir, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);

	ierr = MatAYPX(lhs, -1, Ir, SAME_NONZERO_PATTERN); CHKERRQ(ierr);

	ierr = lapackMatInv(lhs); CHKERRQ(ierr);

	ierr = MatMatMult(lhs, Atilde, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Gtilde); CHKERRQ(ierr);

#ifdef DEBUG_DMD
	ierr = printMatPYTHON(DEB_MAT_DIR + "Ir", "Ir", Ir); CHKERRQ(ierr);
	ierr = printMatPYTHON(DEB_MAT_DIR + "lhs_inv", "lhs_inv", lhs); CHKERRQ(ierr);
	ierr = printMatPYTHON(DEB_MAT_DIR + "full", "full", Gtilde); CHKERRQ(ierr);
#endif
#ifdef CALC_CONDITION_NUMBER_OF_UPDATE
	SVD svd;
	_svd AtildeSVD;
	bool squreMat = true;
	ierr = solveSVD(svd, Atilde); CHKERRQ(ierr);
	ierr = calcLowRankSVDApprox(svd, svdRank, AtildeSVD, "Atilde", squreMat); CHKERRQ(ierr);

	ierr = SVDReset(svd); CHKERRQ(ierr);
	ierr = solveSVD(svd, lhs); CHKERRQ(ierr);
	ierr = calcLowRankSVDApprox(svd, svdRank, AtildeSVD, "(I-Atilde)", squreMat); CHKERRQ(ierr);

	ierr = SVDDestroy(&svd); CHKERRQ(ierr);
#endif

	Mat UrGt{};
	Vec X2_end_tilde{};
	ierr = VecCreateSeq(PETSC_COMM_SELF, X.num_rows, &update); CHKERRQ(ierr);
	ierr = VecCreateSeq(PETSC_COMM_SELF, svdRank, &X2_end_tilde); CHKERRQ(ierr);

	auto start = std::chrono::steady_clock::now();
	ierr = MatGetColumnVector(X2_tilde, X2_end_tilde, X.num_cols - 2); CHKERRQ(ierr);
	// UrGt = Ur * Gtilde
	ierr = MatMatMult(lrSVD.Ur, Gtilde, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &UrGt); CHKERRQ(ierr);
	// UrGt * X2_end_tilde
	ierr = MatMult(UrGt, X2_end_tilde, update);
	std::string sMessage = "DMD - computing the update (matrix mult.) time";
	recordTime(start, sMessage);

	ierr = MatDestroy(&Ir); CHKERRQ(ierr);
	ierr = MatDestroy(&lhs); CHKERRQ(ierr);
	ierr = MatDestroy(&Gtilde); CHKERRQ(ierr);
	ierr = MatDestroy(&UrGt); CHKERRQ(ierr);
	ierr = VecDestroy(&X2_end_tilde); CHKERRQ(ierr);

	return ierr;
}

PetscErrorCode DMD::computeUpdate(PetscInt iMode){
	PetscErrorCode ierr;
	PetscScalar TDend, slope, conv_sum;

	assert(iMode < svdRank
			&& "Requesting a mode that does not exist!! Did you set -dmd_rank correctly?");
	slope = std::real(lrSVD.eigs[iMode]);

	ierr = VecCreateSeq(PETSC_COMM_SELF, X.num_rows, &update); CHKERRQ(ierr);

	ierr = MatGetValue(time_dynamics, X.num_cols - 1, iMode, &TDend); CHKERRQ(ierr);
	conv_sum = slope * TDend / (1 - slope);

	ierr = MatGetColumnVector(Phi, update, iMode); CHKERRQ(ierr);
	ierr = VecScale(update, conv_sum); CHKERRQ(ierr);

	return ierr;
}

//PetscErrorCode DMD::applyDMD(){
//	PetscErrorCode ierr;
//
//	ierr = prepareData();CHKERRQ(ierr);
//	ierr = regression();CHKERRQ(ierr);
//	ierr = calcDMDmodes();CHKERRQ(ierr);
//
//	for(int i = 0; i < iNumModes; i++){
//		ierr = computeUpdate(i);CHKERRQ(ierr);
//	}
//
//	/* counting the number of calls to this class */
//	isDMD_execs++;
//	return ierr;
//}

PetscErrorCode DMD::applyDMDMatTrans() {
	PetscErrorCode ierr;

	printMatMATLAB("data", "data", *X.mat);

	auto start = std::chrono::steady_clock::now();
	ierr = prepareData();CHKERRQ(ierr);
	std::string sMessage = "DMD - Splitting the snapshots time";
	recordTime(start, sMessage);

	start = std::chrono::steady_clock::now();
	ierr = regression();CHKERRQ(ierr);
	sMessage = "DMD - SVD and computing the best-fit time:";
	recordTime(start, sMessage);

//	ierr = calcDMDmodes();CHKERRQ(ierr);

	start = std::chrono::steady_clock::now();
	ierr = computeMatTransUpdate();CHKERRQ(ierr);
	sMessage = "DMD - computeMatTransUpdate() time:";
	recordTime(start, sMessage);

	fprintf(fLog, "----Norm of our approximation: %e-----\n", dUpdateNorm);

	/* counting the number of calls to this class */
	isDMD_execs++;
	return ierr;
}

PetscErrorCode DMD::DummyDMD(){
	PetscErrorCode ierr;
	DMD::isDMD_execs = 0;
	printMatMATLAB("DumData", "data", *X.mat);

	ierr = prepareData(); CHKERRQ(ierr);
	ierr = regression(true); CHKERRQ(ierr);

	return ierr;
}

PetscErrorCode DMD::solveSVD(SVD& svd, Mat& mMatrix){
	PetscErrorCode ierr;
	PetscInt nsv{};

	ierr = MatGetSize(mMatrix, NULL, &nsv); CHKERRQ(ierr);

	ierr = SVDCreate(PETSC_COMM_WORLD, &svd);
	CHKERRQ(ierr);
	ierr = SVDSetOperator(svd, mMatrix); // This function is changed to SVDSetOperators(SVD svd,Mat A,Mat B) in the new version of SLEPC.
	CHKERRQ(ierr);
	ierr = SVDSetType(svd, SVDLAPACK);
	CHKERRQ(ierr);
	ierr = SVDSetWhichSingularTriplets(svd, SVD_LARGEST);
	CHKERRQ(ierr);
	ierr = SVDSetDimensions(svd, nsv, PETSC_DEFAULT, PETSC_DEFAULT);
	CHKERRQ(ierr);
	ierr = SVDSetFromOptions(svd);
	CHKERRQ(ierr);

	ierr = PetscPrintf(PETSC_COMM_WORLD,
				"\nSolving the Singular Value Decomposition problem...\n");
		CHKERRQ(ierr);
	ierr = SVDSolve(svd);
	CHKERRQ(ierr);

#ifdef DEBUG_DMD
	PetscReal tol;
	PetscInt ncv, maxit, nconv { 0 };
	//gets number of singular values to compute and dimension of the subspace
	ierr = SVDGetDimensions(svd, &nsv, &ncv, NULL);
	CHKERRQ(ierr);
	ierr = SVDGetTolerances(svd, &tol, &maxit);
	CHKERRQ(ierr);
	ierr = SVDGetConverged(svd, &nconv);
	CHKERRQ(ierr);

	std::printf("Number of requested singular values (nsv) = %d,   ncv = %d\n",
			nsv, ncv);
	std::printf("Stopping condition: tol=%.4g, maxit=%d\n", tol, maxit);
	std::printf("Number of converged singular values: %d\n", nconv);
	std::printf("\n");
#endif

	return ierr;
}

/*
 * Compute the proper rank for the matrix trasformation update - Automatically
 */
PetscErrorCode DMD::computeSVDRank(SVD &svd) {
	PetscErrorCode ierr;
	PetscInt nconv, iRankCount{};
	PetscReal sigma;
	std::vector<PetscReal> sigmaVec{};

	ierr = SVDGetConverged(svd, &nconv); CHKERRQ(ierr);

	for (int j = 0; j < nconv; j++) {
		ierr = SVDGetSingularTriplet(svd, j, &sigma, NULL, NULL); CHKERRQ(ierr);
		sigmaVec.push_back(sigma);

		if (sigma > 1e-15)
			iRankCount++;
	}

	svdRank = iRankCount;

	return ierr;
}


/* Extracts singularvalues and singular vectors of a given matrix */
PetscErrorCode DMD::calcLowRankSVDApprox(SVD &svd, PetscInt rank, _svd &LowSVD, std::string sFileName,
		bool squreMat) {
	PetscErrorCode ierr;

	Vec u, v;
	PetscReal sigma;
	PetscInt numCols{rank};
	PetscInt *VrCol_index;
	PetscScalar *column_vec, *row_vec;

	PetscInt X1Rows { X.num_rows };
	PetscInt X1Cols { X.num_cols - 1 };

	if (squreMat) {
		X1Rows = rank;
		X1Cols = rank;
	}

	ierr = VecCreateSeq(PETSC_COMM_WORLD, X1Rows, &u); CHKERRQ(ierr);
	ierr = VecCreateSeq(PETSC_COMM_WORLD, X1Cols, &v); CHKERRQ(ierr);

	ierr = PetscMalloc1(X1Cols, &column_vec); CHKERRQ(ierr);
	ierr = PetscMalloc1(X1Rows, &row_vec); CHKERRQ(ierr);
	ierr = PetscMalloc1(X1Cols, &VrCol_index);

	for (int i = 0; i < X1Cols; i++){
		VrCol_index[i] = i;
	}

	ierr = MatDestroy(&LowSVD.Ur);	CHKERRQ(ierr);
	ierr = MatDestroy(&LowSVD.Sr);	CHKERRQ(ierr);
	ierr = MatDestroy(&LowSVD.Vr);	CHKERRQ(ierr);

	ierr = MatCreate(PETSC_COMM_WORLD, &LowSVD.Ur); CHKERRQ(ierr);
	ierr = MatCreate(PETSC_COMM_WORLD, &LowSVD.Sr); CHKERRQ(ierr);
	ierr = MatCreate(PETSC_COMM_WORLD, &LowSVD.Vr);CHKERRQ(ierr);

	ierr = MatSetSizes(LowSVD.Ur, PETSC_DECIDE, PETSC_DECIDE, X1Rows, numCols);
	CHKERRQ(ierr);
	ierr = MatSetSizes(LowSVD.Sr, PETSC_DECIDE, PETSC_DECIDE, numCols, numCols);
	CHKERRQ(ierr);
	ierr = MatSetSizes(LowSVD.Vr, PETSC_DECIDE, PETSC_DECIDE, X1Cols, numCols);
	CHKERRQ(ierr);

	ierr = MatSetType(LowSVD.Ur, MATAIJ);
	CHKERRQ(ierr);
	ierr = MatSetType(LowSVD.Sr, MATAIJ); CHKERRQ(ierr);
	ierr = MatSetType(LowSVD.Vr, MATAIJ); CHKERRQ(ierr);

	ierr = MatSetUp(LowSVD.Ur);	CHKERRQ(ierr);
	ierr = MatSetUp(LowSVD.Sr);	CHKERRQ(ierr);
	ierr = MatSetUp(LowSVD.Vr);	CHKERRQ(ierr);

	ierr = MatZeroEntries(LowSVD.Ur); CHKERRQ(ierr);
	ierr = MatZeroEntries(LowSVD.Sr); CHKERRQ(ierr); //zeros all entries of a matrix
	ierr = MatZeroEntries(LowSVD.Vr); CHKERRQ(ierr);

		/* Getting SVD modes column by column */
	std::ofstream out;
	out.open(sFileName + "-Sr_vec.dat");
	PetscReal sig1, sig2, sigRate;

	for (int j = 0; j < numCols; j++) {
		PetscInt index = j;
		ierr = SVDGetSingularTriplet(svd, j, &sigma, u, v); CHKERRQ(ierr);

		if (j > 0 && j < numCols) {
			sig2 = sigma;
			sigRate = sig2 / sig1;
			out << sigma << "\t" << sigRate << std::endl;
		}
		if (j < numCols - 1) {
			sig1 = sigma;
			if (j == 0) {
				out << sigma << "\t" << 0 << std::endl;
			}
		}

		ierr = MatSetValue(LowSVD.Sr, j, j, sigma, INSERT_VALUES);	CHKERRQ(ierr);
		ierr = VecGetValues(u, X1Rows, row_index, row_vec);		CHKERRQ(ierr);
		ierr = MatSetValues(LowSVD.Ur, X1Rows, row_index, 1, &index, row_vec, INSERT_VALUES); CHKERRQ(ierr);
		ierr = VecGetValues(v, X1Cols, VrCol_index, column_vec); CHKERRQ(ierr);
		ierr = MatSetValues(LowSVD.Vr, X1Cols, VrCol_index, 1, &index, column_vec, INSERT_VALUES); CHKERRQ(ierr);
	}

	ierr = MatAssemblyBegin(LowSVD.Ur, MAT_FINAL_ASSEMBLY);	CHKERRQ(ierr);
	ierr = MatAssemblyEnd(LowSVD.Ur, MAT_FINAL_ASSEMBLY);	CHKERRQ(ierr);
	ierr = MatAssemblyBegin(LowSVD.Sr, MAT_FINAL_ASSEMBLY);	CHKERRQ(ierr);
	ierr = MatAssemblyEnd(LowSVD.Sr, MAT_FINAL_ASSEMBLY);	CHKERRQ(ierr);
	ierr = MatAssemblyBegin(LowSVD.Vr, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(LowSVD.Vr, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);


#ifdef DMD_SIGMARATIO
	PetscScalar sigma1, sigma2, sigmaMax;
	ierr = MatGetValue(LowSVD.Sr, 0, 0, &sigma1); CHKERRQ(ierr);
	if (!squreMat) {
		ierr = MatGetValue(LowSVD.Sr, 1, 1, &sigma2); CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD, "Sigma ratio is: %3f\n", sigma1/sigma2); CHKERRQ(ierr);
		fprintf(fLog, "%7s \u03C31/\u03C32: %3f\n", sFileName.c_str(), sigma1/sigma2);

	} else {
		ierr = MatGetValue(LowSVD.Sr, numCols - 1, numCols -1, &sigmaMax); CHKERRQ(ierr);
		PetscScalar condNumb = sigma1 / sigmaMax;
		out << "# Matrix condition number: " << condNumb << std::endl;
		ierr = PetscPrintf(PETSC_COMM_WORLD, "%s condition number is: %3f\n", sFileName.c_str(), condNumb); CHKERRQ(ierr);
		fprintf(fLog, "\u03BA(%s): %.3f\n", sFileName.c_str(), condNumb);
	}
#endif
	out.close();

#ifdef DEBUG_DMD
	printMatPYTHON(DEB_TOOL_DIR + sFileName + "-Ur", "Ur", LowSVD.Ur);
	printMatMATLAB(DEB_TOOL_DIR + sFileName + "-Sr", "Sr", LowSVD.Sr);
	printMatPYTHON(DEB_TOOL_DIR + sFileName + "-Vr", "Vr", LowSVD.Vr);
#endif

	PetscFree(column_vec);
	PetscFree(row_vec);

	ierr = VecDestroy(&u); CHKERRQ(ierr);
	ierr = VecDestroy(&v); CHKERRQ(ierr);

	return ierr;
}

PetscErrorCode DMD::calcBestFitlrSVD(_svd& LowSVD, Mat& mBestFit){
	PetscErrorCode ierr;
	Vec vDiagSr;
	Mat UrT_X2;
	PetscInt rank;

	ierr = MatGetSize(LowSVD.Sr, &rank, NULL); CHKERRQ(ierr);

	assert(rank >= 1
			&& "MATRIX INVERSION: SVD rank is less than one. Have you defined it?");
	// Invert Sr matrix - reciprocal of the diagonal
	ierr = VecCreateSeq(PETSC_COMM_SELF, rank, &vDiagSr); CHKERRQ(ierr);
	ierr = MatGetDiagonal(LowSVD.Sr, vDiagSr); CHKERRQ(ierr);
	ierr = VecReciprocal(vDiagSr); CHKERRQ(ierr);
	ierr = MatDuplicate(LowSVD.Sr, MAT_COPY_VALUES, &LowSVD.Sr_inv); CHKERRQ(ierr);
	ierr = MatDiagonalSet(LowSVD.Sr_inv, vDiagSr, INSERT_VALUES); CHKERRQ(ierr);

	// Constructing the best-fit regression of A (Atilde)
	ierr = MatTransposeMatMult(LowSVD.Ur, X2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &UrT_X2); CHKERRQ(ierr);
	ierr = MatMatMatMult(UrT_X2, LowSVD.Vr, LowSVD.Sr_inv, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &mBestFit); CHKERRQ(ierr);

	ierr = MatDestroy(&UrT_X2); CHKERRQ(ierr);
	ierr = VecDestroy(&vDiagSr); CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode DMD::calcUpdateNorm(const _svd &LowSVD, const Mat &mAtilde,
		const Mat &mX1, const Mat &mX2) {
	PetscErrorCode ierr;
	// Computing the norm of our truncation, by taking everything to low-dimensional subspace
	auto start = std::chrono::steady_clock::now();
	Mat X2t_copy, X1_tilde;
	ierr = MatTransposeMatMult(LowSVD.Ur, mX1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &X1_tilde); CHKERRQ(ierr);
	ierr = MatMatMult(mAtilde, X1_tilde, MAT_REUSE_MATRIX, PETSC_DEFAULT, &X1_tilde); CHKERRQ(ierr);
	ierr = MatTransposeMatMult(LowSVD.Ur, mX2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &X2_tilde); CHKERRQ(ierr);
	ierr = MatDuplicate(X2_tilde, MAT_COPY_VALUES, &X2t_copy); CHKERRQ(ierr);
	ierr = MatAXPY(X2t_copy, -1, X1_tilde, SAME_NONZERO_PATTERN); CHKERRQ(ierr);

	ierr = MatNorm(X2t_copy, NORM_FROBENIUS, &dUpdateNorm); CHKERRQ(ierr);

	std::string sMessage = "DMD - Computing the Norm time";
	recordTime(start, sMessage);

	ierr = MatDestroy(&X2t_copy); CHKERRQ(ierr);
	ierr = MatDestroy(&X1_tilde); CHKERRQ(ierr);

	return ierr;
}


/*
 * This function does not consider complex numbers in the eigenvectors matrix - SHOULD BE FIXED
 */

PetscErrorCode DMD::calcEigenvalues(_svd& LowSVD, Mat& matrix, std::string sFileName, bool calcEigenvectors) {
	PetscErrorCode ierr;
	PetscInt nconv;
	EPS eps;
	PetscInt rank;
	ComplexSTLVec omega;

	ierr = MatGetSize(matrix, &rank, NULL); CHKERRQ(ierr);

	/* -----Solving eigenvalue problem ---*/
	ierr = EPSCreate(PETSC_COMM_WORLD, &eps); CHKERRQ(ierr);
	ierr = EPSSetOperators(eps, matrix, PETSC_NULL); CHKERRQ(ierr);
	ierr = EPSSetProblemType(eps, EPS_NHEP); CHKERRQ(ierr);
	ierr = EPSSetType(eps, EPSLAPACK); CHKERRQ(ierr);
	ierr = EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL); CHKERRQ(ierr);
	ierr = EPSSetDimensions(eps, rank, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
	ierr = EPSSetTolerances(eps, 1e-10, 10); CHKERRQ(ierr);
	ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);

	ierr = EPSSolve(eps);	CHKERRQ(ierr);
	ierr = EPSGetConverged(eps, &nconv); CHKERRQ(ierr);

#ifdef DEBUG_DMD_EPS
	/*
	 * Get some information from the solver and display it
	 */
	PetscReal tol;
	PetscInt nev, maxit, its;
	EPSType type;
		ierr = EPSGetDimensions(eps, &nev, NULL, NULL);
		CHKERRQ(ierr);
		ierr = EPSGetTolerances(eps, &tol, &maxit);
		CHKERRQ(ierr);

		ierr = EPSGetIterationNumber(eps, &its);
		CHKERRQ(ierr);
		ierr = EPSGetType(eps, &type);
		CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD, "Solution method: %s\n",
				type);
		CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD,
				"Number of iterations of the method: %d\n", its);
		CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD,
				"Number of requested eigenvalues (nev) = %d\n", nev);
		CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD,
				"Stopping condition: tol=%.4g, maxit=%d\n", tol,
				maxit);
		CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD,
				"# Number of converged eigenpairs: %d\n\n", nconv);
		CHKERRQ(ierr);
#endif

	/* ------ Extracting eigenvalues --------- */
	PetscScalar dReal, dImag, dError;

	// For calcmodes
#ifdef COMPLEX_NUMBER_PROBLEM
	Vec vr, vi;
	if (calcEigenvectors){
		ierr = MatCreateVecs(matrix, NULL, &vr); CHKERRQ(ierr);
		ierr = MatCreateVecs(matrix, NULL, &vi); CHKERRQ(ierr);

		ierr = MatDestroy(&LowSVD.W); CHKERRQ(ierr);
		ierr = MatCreate(PETSC_COMM_WORLD, &LowSVD.W); CHKERRQ(ierr);
		ierr = MatSetSizes(LowSVD.W, PETSC_DECIDE, PETSC_DECIDE, rank, rank); CHKERRQ(ierr);
		ierr = MatSetType(LowSVD.W, MATAIJ); CHKERRQ(ierr);
		ierr = MatSetUp(LowSVD.W); CHKERRQ(ierr);
		ierr = MatZeroEntries(LowSVD.W); CHKERRQ(ierr);
	}
#endif

	FILE *fEigs;
	std::string sName = DEB_TOOL_DIR + sFileName + "-TransMatrix.dat";
	fEigs = fopen(sName.data(), "w");

	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nExtracting %s...\n", sFileName.c_str()); CHKERRQ(ierr);

	std::fprintf(fEigs, "VARIABLES = \"NUM\" \"sigma-Real\" \"sigma-Imag\" \"Rel. Error\" \"Eig-Real\""
			"\"Eig-Imag\"\n");

	for (PetscInt i = 0; i < nconv; i++) {

//		if (calcEigenvectors) {
//			ierr = EPSGetEigenpair(eps, i, &dReal, &dImag, vr, vi);	CHKERRQ(ierr);
//		} else {
		// actually, amplification factor
		ierr = EPSGetEigenvalue(eps, i, &dReal, &dImag);CHKERRQ(ierr);
		ierr = EPSComputeError(eps, i, EPS_ERROR_RELATIVE, &dError);CHKERRQ(ierr);
		LowSVD.eigs.push_back({dReal, dImag});

		// Computing omega - solutions eigenvalues
		ComplexNum tmp2 = log(LowSVD.eigs[i]);
		assert(isfinite(std::real(tmp2)) && "Omega has infinite value!!\n\n");
		omega.push_back(tmp2 / dt);
		double OmegaReal = omega[i].real();
		double OmegaImag = omega[i].imag();

		std::fprintf(fEigs, "%.2i\t%.12g\t%12g\t%.12g\t%.12g\t%.12g\t\n", i + 1,
				dReal, dImag, dError, OmegaReal, OmegaImag);

#ifdef PRINT_EIGENVALUES
		printf("eigen %i: %f %fi\n", i + 1, std::real(LowSVD.eigs[i]), std::imag(LowSVD.eigs[i]));
#endif
#ifdef COMPLEX_NUMBER_PROBLEM
		PetscScalar dVecr, dVeci;

		if (calcEigenvectors) {
			for (int row = 0; row < rank; row++) {
				/* Get values one-by-one and write them one at a time */
				ierr = VecGetValues(vr, 1, &row, &dVecr);
				CHKERRQ(ierr);
				ierr = VecGetValues(vi, 1, &row, &dVeci);
				CHKERRQ(ierr);
				/* Setting the absolute value */
//				value = std::sqrt(dVecr * dVecr + dVeci * dVeci);
				/* Setting the real part */
				ComplexNum cmpxValue(dVecr, dVeci);
				PetscScalar value = cmpxValue;
				ierr = MatSetValue(LowSVD.W, row, i, value, INSERT_VALUES);
				CHKERRQ(ierr);
			}
		}
#endif
	}
#ifdef COMPLEX_NUMBER_PROBLEM
	if (calcEigenvectors) {
		ierr = MatAssemblyBegin(LowSVD.W, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
		ierr = MatAssemblyEnd(LowSVD.W, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	}
	ierr = printMatPYTHON("PHI", "PHI", LowSVD.W); CHKERRQ(ierr);
#endif

	std::fprintf(fEigs, "\n");
	std::fclose(fEigs);

	ierr = EPSDestroy(&eps);CHKERRQ(ierr);
#ifdef COMPLEX_NUMBER_PROBLEM
	if (calcEigenvectors){
		ierr = VecDestroy(&vr); CHKERRQ(ierr);
		ierr = VecDestroy(&vi); CHKERRQ(ierr);
	}
#endif

	return ierr;
}

PetscErrorCode DMD::calcDominantModePeriod(Mat& matrix) {
	PetscErrorCode ierr;
	EPS eps;
	PetscInt rank, nconv;

	ierr = MatGetSize(matrix, &rank, NULL); CHKERRQ(ierr);

	/* -----Solving eigenvalue problem ---*/
	ierr = EPSCreate(PETSC_COMM_WORLD, &eps); CHKERRQ(ierr);
	ierr = EPSSetOperators(eps, matrix, PETSC_NULL); CHKERRQ(ierr);
	ierr = EPSSetProblemType(eps, EPS_NHEP); CHKERRQ(ierr);
	ierr = EPSSetType(eps, EPSLAPACK); CHKERRQ(ierr);
	ierr = EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL); CHKERRQ(ierr);
	ierr = EPSSetDimensions(eps, rank, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
	ierr = EPSSetTolerances(eps, 1e-10, 10); CHKERRQ(ierr);
	ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);

	ierr = EPSSolve(eps);	CHKERRQ(ierr);
	ierr = EPSGetConverged(eps, &nconv); CHKERRQ(ierr);


	/* ------ Extracting eigenvalues --------- */
	PetscScalar dReal, dImag;

	for (PetscInt i = 0; i < nconv; i++) {
		ierr = EPSGetEigenvalue(eps, i, &dReal, &dImag);CHKERRQ(ierr);
		printf("eigen %i: %f %fi\n", i + 1, dReal, dImag);
	}

	ierr = EPSGetEigenvalue(eps, 0, &dReal, &dImag); CHKERRQ(ierr);
	printf ("\nimag: %f\n", dImag);
	if (abs(dImag) <= 1e-15) {
		ierr = PetscPrintf(PETSC_COMM_WORLD,
				"\nCalc Dominant Mode Period: Dominant eigenvalue is Real\n");	CHKERRQ(ierr);
		iOsclPeriod = 0;
		return ierr;
	}
	ComplexNum dominantEig(dReal, dImag);
	ComplexNum tmp2 = log(dominantEig)/dt;
	PetscReal EigImag = std::imag(tmp2);
	iOsclPeriod = std::ceil(M_PIl / (dt * EigImag));

	ierr = EPSDestroy(&eps);CHKERRQ(ierr);
	return ierr;
}


PetscErrorCode DMD::lapackMatInv(Mat &A){
	PetscErrorCode ierr;
	PetscInt nRows, nCols;

	ierr = MatGetSize(A, &nRows, &nCols); CHKERRQ(ierr);
	assert(nRows == nCols && "Trying to compute inverse of a non-square matrix!!");
	int N = nRows;

	std::vector<PetscScalar> dMat{};
	int* IPIV = new int[N];

	for (int i = 0; i < nRows; i++)
		for (int j = 0; j < nCols; j++) {
			PetscScalar value;
			ierr = MatGetValue(A, i, j, &value); CHKERRQ(ierr);
			dMat.push_back(value);
		}

	ierr = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, N, N, dMat.data(), N, IPIV);
	if (ierr != 0) {
		printf("\nError computing the LU factorization using LAPCKE!!");
		return ierr;
	}

	ierr = LAPACKE_dgetri(LAPACK_ROW_MAJOR, N, dMat.data(), N, IPIV);
	if (ierr != 0) {
		printf("\nError in computing the inverse using LAPACKE!!");
		return ierr;
	}

	for(int i = 0, index = 0; i < nRows; i++)
		for (int j = 0; j < nCols; j++, index++) {
			PetscScalar value = dMat[index];
			ierr = MatSetValue(A, i, j, value, INSERT_VALUES); CHKERRQ(ierr);
		}

	ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

	delete[] IPIV;

	return ierr;
}


PetscErrorCode DMD::printVecMATLAB(std::string sFileName,
		std::string sVectorName, Vec V) const {
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

PetscErrorCode DMD::printMatMATLAB(std::string sFilename,
		std::string sMatrixName, Mat A) const {
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

PetscErrorCode DMD::printVecPYTHON(std::string sFileName,
		std::string sVectorName, Vec V) const {
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

PetscErrorCode DMD::printMatPYTHON(std::string sFilename,
		std::string sMatrixName, Mat A) const {
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

void DMD::recordTime(std::chrono::steady_clock::time_point start,
		std::string sMessage){
	auto stop =	std::chrono::steady_clock::now();
	std::chrono::duration<double> duration = stop - start;
//	std::printf("%s: %f [seconds]\n", sMessage.c_str(), duration.count());
	fprintf(fLog, "%s: %f [seconds]\n", sMessage.c_str(), duration.count());
}



