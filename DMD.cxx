/*
 * DMD.cpp
 *
 *  Created on: Jul 30, 2022
 *      Author: mirshahi
 */

#include "DMD.h"

int DMD::isDMD_execs = 0;

DMD::DMD(const Mat *svdMat, PetscInt iModes, PetscReal DT) :
	iNumModes(iModes), dt(DT) {

	/* Opening a log file to write some results, to access after code execution */
	fLog = fopen("Log.dat", "a");

	/* Set the snapshots matrix and get the sizes of its rows and columns */
	X.mat = svdMat;
	MatGetSize(*X.mat, &X.num_rows, &X.num_cols);
	PetscMalloc1(X.num_rows, &row_index);

	for (int i = 0; i < X.num_rows; i++) {
		row_index[i] = i;
	}

	/* Get the number of modes to eliminate at each call to the DMD class */
	PetscBool flg;
	PetscInt iDMD, *ipDMDRanks;

	PetscOptionsGetInt(NULL, NULL, "-DMD_nits", &iDMD, &flg);
	PetscMalloc1(iDMD, &ipDMDRanks);

	PetscOptionsGetIntArray(PETSC_NULL, PETSC_NULL, "-DMD_ranks", ipDMDRanks, &iDMD, &flg);
	if (!flg) {
		PetscErrorPrintf("Missing -dmd_ranks flag!\n");
		exit(2);
	} else if (svdRank > X.num_cols - 1) {
		PetscErrorPrintf("\nSVD rank is greater than the number of (snapshots columns - 1).\n"
				"Decrease the SVD rank, or increase the number of snapshots columns.\n");
		exit(2);
	}
	svdRank = ipDMDRanks[isDMD_execs];
	printf("\nDMD was called %i times, current SVD approximation rank: %i\n", isDMD_execs + 1, svdRank);

	/* Writing information to our log file */
	fprintf(fLog, "SVDRank: %i\t num Snapshots: %i-1\n", svdRank, X.num_cols);

	/* creating a few directories */
	if (!IsPathExist(DEB_MAT_DIR)) {
		if (mkdir(DEB_MAT_DIR.c_str(), 0777) == -1)
			PetscErrorPrintf("Error : %s \n", strerror(errno));
		else
			PetscPrintf(PETSC_COMM_WORLD, "%s created", DEB_MAT_DIR);
	}
	if (!IsPathExist(DEB_TOOL_DIR)) {
		if (mkdir(DEB_TOOL_DIR.c_str(), 0777) == -1)
			PetscErrorPrintf("Error : %s \n", strerror(errno));
		else
			PetscPrintf(PETSC_COMM_WORLD, "%s created", DEB_TOOL_DIR);
	}

}



DMD::~DMD() {
	fprintf(fLog, "\n");
	fclose(fLog);
	PetscFree(row_index);
	MatDestroy(&X1);
	MatDestroy(&X2);
	MatDestroy(&lrSVD.Ur);
	MatDestroy(&lrSVD.Sr);
	MatDestroy(&lrSVD.Vr);
	MatDestroy(&lrSVD.Sr_inv);
	MatDestroy(&lrSVD.W);
	MatDestroy(&Atilde);
	MatDestroy(&Phi);
	MatDestroy(&time_dynamics);
	VecDestroy(&update);

	/* counting the number of calls to this class */
	isDMD_execs++;
}


/*
 * Splitting data into 2 matrices
 */
PetscErrorCode DMD::prepareData(){
	PetscErrorCode ierr;
	PetscInt *X1col_index, *X2col_index;
	PetscScalar *dTmpArr;
	PetscInt cols{X.num_cols}, rows{X.num_rows};

	ierr = PetscMalloc1(cols - 1, &X1col_index); CHKERRQ(ierr);
	ierr = PetscMalloc1(cols - 1, &X2col_index); CHKERRQ(ierr);
	ierr = PetscMalloc1(rows*(cols - 1), &dTmpArr); CHKERRQ(ierr);


	for (int i = 0; i < cols - 1; i++) {
		X1col_index[i] = i; //X1col_index is used for both X1 and X2 for setting values
		X2col_index[i] = i + 1;
	}

//	ierr = PetscPrintf(PETSC_COMM_WORLD,
//			"Preparing matrices for Dynamic mode decomposition...\n");	CHKERRQ(ierr);

	ierr = MatDestroy(&X1);	CHKERRQ(ierr);
	ierr = MatDestroy(&X2);	CHKERRQ(ierr);

	ierr = MatCreate(MPI_COMM_WORLD, &X1);	CHKERRQ(ierr);
	ierr = MatSetSizes(X1, PETSC_DECIDE, PETSC_DECIDE, rows,
			cols - 1);	CHKERRQ(ierr);
	ierr = MatSetType(X1, MATAIJ); CHKERRQ(ierr);
	ierr = MatSetUp(X1);	CHKERRQ(ierr);

	ierr = MatCreate(MPI_COMM_WORLD, &X2);	CHKERRQ(ierr);
	ierr = MatSetSizes(X2, PETSC_DECIDE, PETSC_DECIDE, rows,
			cols - 1);	CHKERRQ(ierr);
	ierr = MatSetType(X2, MATAIJ); CHKERRQ(ierr);
	ierr = MatSetUp(X2);	CHKERRQ(ierr);

	ierr = MatGetValues(*X.mat, rows, row_index, cols - 1, X1col_index, dTmpArr);
	ierr = MatSetValues(X1, rows, row_index, cols - 1, X1col_index, dTmpArr,
			INSERT_VALUES);
	CHKERRQ(ierr);

	ierr = MatAssemblyBegin(X1, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);
	ierr = MatAssemblyEnd(X1, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);

	ierr = MatGetValues(*X.mat, rows, row_index, cols - 1, X2col_index, dTmpArr);
	ierr = MatSetValues(X2, rows, row_index, cols - 1, X1col_index, dTmpArr,
			INSERT_VALUES);
	CHKERRQ(ierr);

	ierr = MatAssemblyBegin(X2, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);
	ierr = MatAssemblyEnd(X2, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);

#ifdef DEBUG_DMD
	printMatPYTHON(DEB_MAT_DIR + "X", "X", *X.mat);
	printMatMATLAB(DEB_MAT_DIR + "X", "X", *X.mat);

	printMatPYTHON(DEB_MAT_DIR + "X1", "X1", X1);
	printMatPYTHON(DEB_MAT_DIR + "X2", "X2", X2);
#endif

	ierr = PetscFree(X1col_index); CHKERRQ(ierr);
	ierr = PetscFree(X2col_index); CHKERRQ(ierr);
	ierr = PetscFree(dTmpArr); CHKERRQ(ierr);

	return ierr;
}

PetscErrorCode DMD::regression() {
	PetscErrorCode ierr;
	SVD svd;

	ierr = solveSVD(svd, X1); CHKERRQ(ierr);

	/*
	 * Getting Singular values and singular vectors
	 * and building low rank truncation matrices
	 */
	ierr = PetscPrintf(PETSC_COMM_WORLD,
					"computing the %i-rank-SVD-approximation of the snapshots matrix.\n", svdRank); CHKERRQ(ierr);
	ierr = calcLowRankSVDApprox(svd, svdRank, lrSVD, "SnapshotsMat-LowRank"); CHKERRQ(ierr);

#ifdef DMD_CHECK_EIGS
	Mat fullAtilde;
	ierr = calcLowRankSVDApprox(svd, X.num_cols - 1, fullSVD, "SnapshotsMat-Full"); CHKERRQ(ierr);
	/* Finds the FULL Atilde using all the columns (all the modes) */
	ierr = calcBestFitlrSVD(fullSVD, fullAtilde); CHKERRQ(ierr);
	/* Calculates the eigenvalues of the FULL Atilde */
	ierr = calcEigenvalues(fullSVD, fullAtilde, "fullAtilde_eigenvalues"); CHKERRQ(ierr);
	MatDestroy(&fullAtilde);
#endif

	ierr = calcBestFitlrSVD(lrSVD, Atilde); CHKERRQ(ierr);
	/* ------------ Eigen analysis of Atilde -------------*/
	ierr = calcEigenvalues(lrSVD, Atilde, "LowRank-Eigenvalues", true); CHKERRQ(ierr);

	return ierr;
}


PetscErrorCode DMD::calcDMDmodes(){
	PetscErrorCode ierr;
	Mat X2_Vr;

	// Calculating Spatial modes
		ierr = MatMatMult(X2, lrSVD.Vr, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &X2_Vr); CHKERRQ(ierr);
		ierr = MatMatMatMult(X2_Vr, lrSVD.Sr_inv, lrSVD.W, MAT_INITIAL_MATRIX, PETSC_DEFAULT,
						&Phi); CHKERRQ(ierr);

		std::vector<PetscReal> omega;
		for (size_t i = 0; i < lrSVD.eigs.size(); i++){
			PetscReal eig_real = std::real(lrSVD.eigs[i]);
			PetscReal tmp = log(eig_real);
			assert(isfinite(tmp) && "Omega has infinite value!!\n\n");
			omega.push_back(tmp/dt);
		}


	Vec rhs, Soln; // rhs = x1, Soln = b
	KSP ksp;
	PC pc;
	ierr = VecCreateSeq(PETSC_COMM_SELF, X.num_rows, &rhs); CHKERRQ(ierr);
	ierr = VecCreateSeq(PETSC_COMM_SELF, svdRank, &Soln); CHKERRQ(ierr);
	ierr = MatGetColumnVector(*X.mat, rhs, 0); CHKERRQ(ierr);

#ifdef DEBUG_DMD
	ierr = printVecMATLAB(DEB_MAT_DIR + "x1", "x1", rhs); CHKERRQ(ierr);
	ierr = printVecMATLAB(DEB_MAT_DIR + "b", "b", Soln); CHKERRQ(ierr);

#endif
	ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);

	ierr = KSPSetOperators(ksp, Phi, Phi); CHKERRQ(ierr);
	ierr = KSPSetType(ksp, KSPLSQR); CHKERRQ(ierr);
	ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
	ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);
	ierr = KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

	//ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

	ierr = KSPSolve(ksp, rhs, Soln); CHKERRQ(ierr);
//	ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

#ifdef DEBUG_DMD
	/*
	 * The Solution of the least squares (b) is related to the eigenvectors and
	 * the fact that we are not including imaginery parts in the KSP solver
	 * So the results are diffferent from Python and MATLAB
	 */
	ierr = printVecMATLAB(DEB_MAT_DIR + "b", "b", Soln); CHKERRQ(ierr);
#endif

	ierr = MatCreate(MPI_COMM_WORLD, &time_dynamics);	CHKERRQ(ierr);
	ierr = MatSetSizes(time_dynamics, PETSC_DECIDE, PETSC_DECIDE, X.num_cols,
			svdRank);	CHKERRQ(ierr);
	ierr = MatSetType(time_dynamics, MATAIJ); CHKERRQ(ierr);
	ierr = MatSetUp(time_dynamics);	CHKERRQ(ierr);

	PetscReal t = 0;
	for (int iter = 0; iter < X.num_cols; iter++) {
		for (int mode = 0; mode < svdRank; mode++) {
			PetscScalar bVal;
			ierr = VecGetValues(Soln, 1, &mode, &bVal); CHKERRQ(ierr);
			PetscScalar value = bVal*exp(omega[mode]*t);

			ierr = MatSetValue(time_dynamics, iter, mode, value, INSERT_VALUES); CHKERRQ(ierr);
		}
		t += dt;
	}
	ierr = MatAssemblyBegin(time_dynamics, MAT_FINAL_ASSEMBLY);	CHKERRQ(ierr);
	ierr = MatAssemblyEnd(time_dynamics, MAT_FINAL_ASSEMBLY);	CHKERRQ(ierr);

#ifdef DEBUG_DMD
	ierr = printMatPYTHON(DEB_MAT_DIR + "TD", "TD", time_dynamics); CHKERRQ(ierr);
#endif

	ierr = MatDestroy(&X2_Vr); CHKERRQ(ierr);
	ierr = VecDestroy(&rhs); CHKERRQ(ierr);
	ierr = VecDestroy(&Soln); CHKERRQ(ierr);
	ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode DMD::computeMatTransUpdate() {
	PetscErrorCode ierr;
	Mat Ir{}, lhs{}, Gtilde{};

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

	ierr = PetscPrintf(PETSC_COMM_WORLD,
				"\nComputing the inverse..\n");
		CHKERRQ(ierr);

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

	Mat mTmp{}, mTmp2{};
	Vec X2_end{};
	ierr = VecCreateSeq(PETSC_COMM_SELF, X.num_rows, &X2_end); CHKERRQ(ierr);
	ierr = VecCreateSeq(PETSC_COMM_SELF, X.num_rows, &update); CHKERRQ(ierr);

	ierr = MatGetColumnVector(X2, X2_end, X.num_cols - 2); CHKERRQ(ierr);
	ierr = MatMatMult(lrSVD.Ur, Gtilde, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &mTmp); CHKERRQ(ierr);
	ierr = MatMatTransposeMult(mTmp, lrSVD.Ur, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &mTmp2); CHKERRQ(ierr);

	ierr = MatMult(mTmp2, X2_end, update); CHKERRQ(ierr);

	ierr = MatDestroy(&Ir); CHKERRQ(ierr);
	ierr = MatDestroy(&lhs); CHKERRQ(ierr);
	ierr = MatDestroy(&Gtilde); CHKERRQ(ierr);
	ierr = MatDestroy(&mTmp); CHKERRQ(ierr);
	ierr = MatDestroy(&mTmp2); CHKERRQ(ierr);
	ierr = VecDestroy(&X2_end); CHKERRQ(ierr);

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

PetscErrorCode DMD::applyDMD(){
	PetscErrorCode ierr;

	ierr = prepareData();
	CHKERRQ(ierr);
	ierr = regression();
	CHKERRQ(ierr);
	ierr = calcDMDmodes();
	CHKERRQ(ierr);

	for(int i = 0; i < iNumModes; i++){
		ierr = computeUpdate(i);
			CHKERRQ(ierr);
	}

	return ierr;
}

PetscErrorCode DMD::applyDMDMatTrans() {
	PetscErrorCode ierr;

	printMatMATLAB("data", "data", *X.mat);

	ierr = prepareData();CHKERRQ(ierr);
	ierr = regression();CHKERRQ(ierr);
	ierr = calcDMDmodes();CHKERRQ(ierr);
	ierr = computeMatTransUpdate();CHKERRQ(ierr);

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
			if (j == 0){
				out << sigma << std::endl;
		}}

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

#ifdef DEBUG_DMD
	printMatMATLAB(DEB_MAT_DIR + "Sr_inv", "Sr_inv", LowSVD.Sr_inv);
#endif
	// Constructing the best-fit regression of A (Atilde)
	ierr = MatTransposeMatMult(LowSVD.Ur, X2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &UrT_X2); CHKERRQ(ierr);
	ierr = MatMatMatMult(UrT_X2, LowSVD.Vr, LowSVD.Sr_inv, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &mBestFit); CHKERRQ(ierr);

#ifdef DEBUG_DMD
	printMatPYTHON(DEB_MAT_DIR + "urtx2", "urtx2", UrT_X2);
	printMatPYTHON(DEB_MAT_DIR + "Atilde", "Atilde", mBestFit);
#endif


	ierr = MatDestroy(&UrT_X2); CHKERRQ(ierr);
	ierr = VecDestroy(&vDiagSr); CHKERRQ(ierr);
	return ierr;
}



PetscErrorCode DMD::calcEigenvalues(_svd& LowSVD, Mat& matrix, std::string sFileName, bool calcEigenvectors) {
	PetscErrorCode ierr;
	PetscInt nconv;
	EPS eps;
	PetscInt rank;

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
	PetscScalar dVecr, dVeci, value;
	PetscScalar dReal, dImag, dError;
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

	FILE *fEigs;
	std::string sName = DEB_TOOL_DIR + sFileName + "-TransMatrix.dat";
	fEigs = fopen(sName.data(), "w");

	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nExtracting %s...\n", sFileName.c_str()); CHKERRQ(ierr);

	std::fprintf(fEigs, "VARIABLES = \"NUM\" \"Real\" \"Imag\" \"Rel. Error\" \"LogValue\"\n");

	for (PetscInt i = 0; i < nconv; i++) {

		if (calcEigenvectors) {
			ierr = EPSGetEigenpair(eps, i, &dReal, &dImag, vr, vi);	CHKERRQ(ierr);
		} else {
			ierr = EPSGetEigenvalue(eps, i, &dReal, &dImag); CHKERRQ(ierr);
		}

		ierr = EPSComputeError(eps, i, EPS_ERROR_RELATIVE, &dError);CHKERRQ(ierr);
		LowSVD.eigs.push_back({dReal, dImag});

#ifdef PRINT_EIGENVALUES
		printf("eigen %i: %f %fi\n", i + 1, std::real(LowSVD.eigs[i]), std::imag(LowSVD.eigs[i]));
#endif
		std::fprintf(fEigs, "%.2i\t%.12g\t%12g\t%.12g\t%.12g\n", i + 1, dReal, dImag, dError, log(dReal));

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
				value = dVecr;
				ierr = MatSetValue(LowSVD.W, row, i, value, INSERT_VALUES);
				CHKERRQ(ierr);
			}
		}
	}
	if (calcEigenvectors) {
		ierr = MatAssemblyBegin(LowSVD.W, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
		ierr = MatAssemblyEnd(LowSVD.W, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	}

	std::fprintf(fEigs, "\n");
	std::fclose(fEigs);

	ierr = EPSDestroy(&eps);CHKERRQ(ierr);
	if (calcEigenvectors){
		ierr = VecDestroy(&vr); CHKERRQ(ierr);
		ierr = VecDestroy(&vi); CHKERRQ(ierr);
	}
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



