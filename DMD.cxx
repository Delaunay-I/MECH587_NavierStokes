/*
 * DMD.cpp
 *
 *  Created on: Jul 30, 2022
 *      Author: mirshahi
 */

#include "DMD.h"
Eigen::MatrixXd pMat_to_eMat_double(const Mat &pMat);
Eigen::VectorXd pVec_to_eVec_double(const Vec &pVec);


int DMD::isDMD_execs = 0;

DMD::DMD(const Mat *svdMat, PetscReal DT, PetscReal dNorm) :
	 dt(DT), iterNorm(dNorm) {

	/* Opening a log file to write some results, to access after code execution */
	fLog = fopen("Log.dat", "a");
	fML.open("ML_dataset.csv", std::ios_base::app);
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
	fML.close();

	PetscFree(row_index);
	MatDestroy(&X1);
	MatDestroy(&X2);
	MatDestroy(&lrSVD.Ur);
	MatDestroy(&lrSVD.Sr);
	MatDestroy(&lrSVD.Vr);
	MatDestroy(&lrSVD.Sr_inv);
	MatDestroy(&lrSVD.W);
	MatDestroy(&Atilde);
	MatDestroy(&time_dynamics);
	VecDestroy(&update);
}


/*
 * Splitting data into 2 matrices
 * Hence the definition of the rows and columns are opposite
 * This function is implemented for a row-major snapshot matrix
 */
PetscErrorCode DMD::prepareData(){
	PetscErrorCode ierr;
	PetscInt *X1row_index, *X2row_index;
	PetscScalar *dTmpArr;
	PetscInt rows{X.num_cols}, cols{X.num_rows};

	ierr = PetscMalloc1(rows - 1, &X1row_index); CHKERRQ(ierr);
	ierr = PetscMalloc1(rows - 1, &X2row_index); CHKERRQ(ierr);
	ierr = PetscMalloc1(cols*(rows - 1), &dTmpArr); CHKERRQ(ierr);
#ifdef DEBUG_DMD
	ierr = printMatMATLAB("X", "X", *X.mat); CHKERRQ(ierr);
#endif

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
#ifdef DEBUG_DMD
	ierr = printMatMATLAB("X1", "X1", X1); CHKERRQ(ierr);
	ierr = printMatMATLAB("X2", "X2", X2); CHKERRQ(ierr);
#endif

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
	std::string sMessage = "DMD::regression::SVD()";
	recordTime(start, sMessage);
#ifdef DEBUG_DMD
	ierr = testSVD(svd); CHKERRQ(ierr);
#endif
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
		ierr = calcLowRankSVDApprox(svd, svdRank, lrSVD, "LowRnk");
		CHKERRQ(ierr);

#ifdef DMD_CHECK_EIGS
	if (!dummyDMD) {
		Mat fullAtilde;
		ierr = calcLowRankSVDApprox(svd, X.num_cols - 1, fullSVD, "Full"); CHKERRQ(ierr);
		/* Finds the FULL Atilde using all the columns (all the modes) */
		ierr = calcBestFitlrSVD(fullSVD, fullAtilde); CHKERRQ(ierr);
		/* Calculates the eigenvalues of the FULL Atilde */
//		ierr = calcEigenvalues(fullSVD, fullAtilde, "FullAtilde"); CHKERRQ(ierr);
		MatDestroy(&fullAtilde);
	}
#endif

	ierr = calcBestFitlrSVD(lrSVD, Atilde); CHKERRQ(ierr);
	/* ------------ Eigen analysis of Atilde -------------*/
	if (!dummyDMD) {
		ierr = calcEigenvalues(lrSVD, Atilde, "LowAtilde", true); CHKERRQ(ierr);
	} else {
		ierr = calcDominantModePeriod(Atilde); CHKERRQ(ierr);
	}

	/* Computing the norm of the DMD approximation */
	ierr = calcUpdateNorm(lrSVD, Atilde, X1, X2); CHKERRQ(ierr);

	ierr = SVDDestroy(&svd); CHKERRQ(ierr);
	return ierr;
}


PetscErrorCode DMD::computeMatUpdate() {
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
	Vec X2_end{}, X2_end_tilde{};
	ierr = VecCreateSeq(PETSC_COMM_SELF, X.num_rows, &update); CHKERRQ(ierr);
	ierr = VecCreateSeq(PETSC_COMM_SELF, X.num_rows, &X2_end); CHKERRQ(ierr);
	ierr = VecCreateSeq(PETSC_COMM_SELF, svdRank, &X2_end_tilde); CHKERRQ(ierr);

	ierr = MatGetColumnVector(X2, X2_end, X.num_cols - 2); CHKERRQ(ierr);
	// X2_end_tilde = Ur^T * X2[:, -1]
	ierr = MatMultTranspose(lrSVD.Ur, X2_end, X2_end_tilde); CHKERRQ(ierr);
	// UrGt = Ur * Gtilde
	ierr = MatMatMult(lrSVD.Ur, Gtilde, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &UrGt); CHKERRQ(ierr);
	// update = UrGt * X2_end_tilde
	ierr = MatMult(UrGt, X2_end_tilde, update);


	ierr = MatDestroy(&Ir); CHKERRQ(ierr);
	ierr = MatDestroy(&lhs); CHKERRQ(ierr);
	ierr = MatDestroy(&Gtilde); CHKERRQ(ierr);
	ierr = MatDestroy(&UrGt); CHKERRQ(ierr);
	ierr = VecDestroy(&X2_end); CHKERRQ(ierr);
	ierr = VecDestroy(&X2_end_tilde); CHKERRQ(ierr);

	return ierr;
}

PetscErrorCode DMD::applyDMDMatTrans() {
	PetscErrorCode ierr;
#ifdef WRITE_SNAP_MAT
	printMatMATLAB("data_i" + std::to_string(isDMD_execs), "data", *X.mat);
#endif
	auto start = std::chrono::steady_clock::now();
	ierr = prepareData();CHKERRQ(ierr);
	std::string sMessage = "DMD::prepareData()";
	recordTime(start, sMessage);

	start = std::chrono::steady_clock::now();
	ierr = regression();CHKERRQ(ierr);
	sMessage = "DMD::regression()";
	recordTime(start, sMessage);

	start = std::chrono::steady_clock::now();
	ierr = computeMatUpdate();CHKERRQ(ierr);
	sMessage = "DMD::computeMatTransUpdate()";
	recordTime(start, sMessage);

	ierr = calcDMDmodes();CHKERRQ(ierr);

	/* counting the number of calls to this class */
	isDMD_execs++;
	return ierr;
}

PetscErrorCode DMD::DummyDMD(){
	PetscErrorCode ierr;
	DMD::isDMD_execs = 0;
//	printMatMATLAB("DummyData", "data", *X.mat);

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
	ierr = SVDSetOperator(svd, mMatrix);  // This function is changed to SVDSetOperators(SVD svd,Mat A,Mat B) in the new version of SLEPC.
	CHKERRQ(ierr);
	ierr = SVDSetType(svd, SVDLANCZOS);
	CHKERRQ(ierr);
	ierr = SVDSetDimensions(svd, nsv, PETSC_DEFAULT, PETSC_DEFAULT);
	CHKERRQ(ierr);
	ierr = SVDSetWhichSingularTriplets(svd, SVD_LARGEST);
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
 * Used with simple DMD automation
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
	out.open(DEB_TOOL_DIR + sFileName + "-Sr-n" + std::to_string(isDMD_execs) + ".dat");
//	regData_S.open(DEB_TOOL_DIR + "SvLast_data_i" + std::to_string(isDMD_execs) + ".dat", std::ios_base::app);
//	regData_sigRate1.open(DEB_TOOL_DIR + "Sv1_data_i" + std::to_string(isDMD_execs) + ".dat", std::ios_base::app);

	PetscReal sigma1, sigLast, sig2, sigRate;


	for (int j = 0; j < numCols; j++) {
		PetscInt index = j;
		ierr = SVDGetSingularTriplet(svd, j, &sigma, u, v); CHKERRQ(ierr);

#ifdef ML
		//Write singularvalues to the ML dataset
		if (fML.is_open()) {
			fML << sigma << ", ";
		}
#endif

		if (j > 0 && j < numCols) {
			sig2 = sigma;
			sigRate = sig2 / sigLast;
			out << sigma << "\t" << sigRate;

			sigRate = sig2 / sigma1;
			out << "\t" << sigRate << std::endl;
		}
		if (j < numCols - 1) {
			sigLast = sigma;
			if (j == 0) {
				sigma1 = sigma;
				out << sigma << "\t" << 0 << "\t" << 0  << std::endl;
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
#ifdef DEBUG_DMD
	printMatMATLAB(DEB_TOOL_DIR + sFileName + "-Ur", "Ur", LowSVD.Ur);
	printMatMATLAB(DEB_TOOL_DIR + sFileName + "-Sr", "Sr", LowSVD.Sr);
	printMatMATLAB(DEB_TOOL_DIR + sFileName + "-Vr", "Vr", LowSVD.Vr);
#endif

	out.close();

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
	Mat X1Copy, X2approx;
	ierr = MatTransposeMatMult(LowSVD.Ur, mX1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &X1Copy); CHKERRQ(ierr);
	ierr = MatMatMatMult(LowSVD.Ur, mAtilde, X1Copy, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &X2approx); CHKERRQ(ierr);
	ierr = MatAYPX(X2approx, -1, mX2, SAME_NONZERO_PATTERN); CHKERRQ(ierr);

	ierr = MatNorm(X2approx, NORM_FROBENIUS, &dFroNorm); CHKERRQ(ierr);
	fprintf(fLog, "DMD norm:\t%e\tSolution norm:\t%e\n", dFroNorm, iterNorm);
	dFroNorm /= iterNorm;
	fprintf(fLog, "Frobenius norm (Normalized):\t%e\n", dFroNorm);

	ierr = MatDestroy(&X1Copy); CHKERRQ(ierr);
	ierr = MatDestroy(&X2approx); CHKERRQ(ierr);

	return ierr;
}

/*
 * Calculating the eigenvalues of the small space -
 * which are the amplification factors of the original problem
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
	Vec vr, vi;
	ierr = MatCreateVecs(matrix, NULL, &vr); CHKERRQ(ierr);
	ierr = MatCreateVecs(matrix, NULL, &vi); CHKERRQ(ierr);

	FILE *fEigs;
	std::string sName = DEB_TOOL_DIR + sFileName + "-eigs_n" + std::to_string(isDMD_execs) + ".dat";
	fEigs = fopen(sName.data(), "w");

	ierr = PetscPrintf(PETSC_COMM_WORLD, "Eigendecomposition of %s...\n", sFileName.c_str()); CHKERRQ(ierr);

	std::fprintf(fEigs, "VARIABLES = \"NUM\" \"sigma-Real\" \"sigma-Imag\" \"Rel. Error\" \"Eig-Real\""
			"\"Eig-Imag\"\n");

	LowSVD.eigVecs_small.resize(rank, rank);
	LowSVD.eigs.resize(rank);
	LowSVD.omega_sorted.resize(rank);

	for (PetscInt i = 0; i < nconv; i++) {

		ierr = EPSGetEigenpair(eps, i, &dReal, &dImag, vr, vi);	CHKERRQ(ierr);
		ierr = EPSComputeError(eps, i, EPS_ERROR_RELATIVE, &dError);CHKERRQ(ierr);
		ComplexNum tmp{dReal, dImag};
		LowSVD.eigs(i) = tmp;

		// Computing omega - solutions eigenvalues
		assert(isfinite(std::real(log(tmp))) && "Omega has infinite value!!\n\n");
		LowSVD.omega_sorted(i) = log(tmp);
		LowSVD.omega_sorted(i) /= dt;

		std::fprintf(fEigs, "%.2i\t%.12g\t%12g\t%.12g\t%.12g\t%.12g\t\n", i + 1,
				dReal, dImag, dError,
				LowSVD.omega_sorted(i).real() , LowSVD.omega_sorted(i).imag());

		Eigen::VectorXd evReal = pVec_to_eVec_double(vr);
		Eigen::VectorXd evImag = pVec_to_eVec_double(vi);

		LowSVD.eigVecs_small.col(i).real() << evReal;
		LowSVD.eigVecs_small.col(i).imag() << evImag;
	}

#ifdef ML
	Eigen::IOFormat CSVFmt(Eigen::FullPrecision, Eigen::DontAlignCols, ", ");
	if (fML.is_open()) {
		fML << LowSVD.eigs.transpose().real().format(CSVFmt) << ", ";
		fML << LowSVD.omega_sorted.transpose().real().format(CSVFmt) << ", ";
	}
#endif

	std::fprintf(fEigs, "\n");
	std::fclose(fEigs);
#ifdef TEST_EIGEN_SYSTEMS
	ierr = testEPS(); CHKERRQ(ierr);
#endif
	ierr = EPSDestroy(&eps);CHKERRQ(ierr);
	ierr = VecDestroy(&vr); CHKERRQ(ierr);
	ierr = VecDestroy(&vi); CHKERRQ(ierr);

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
	ierr = PetscObjectSetName((PetscObject) V, sVectorName.c_str()); CHKERRQ(ierr);
	ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, sName.c_str(), &viewer); CHKERRQ(ierr);
	ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB); CHKERRQ(ierr);
	ierr = VecView(V, viewer);	CHKERRQ(ierr);
	ierr = PetscViewerPopFormat(viewer); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

	return ierr;
}

PetscErrorCode DMD::printMatMATLAB(std::string sFilename,
		std::string sMatrixName, Mat A) const {
	PetscErrorCode ierr;

	std::string sName = sFilename + ".m";
	PetscViewer viewer;
	ierr = PetscObjectSetName((PetscObject) A, sMatrixName.c_str()); CHKERRQ(ierr);
	ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, sName.c_str(), &viewer); CHKERRQ(ierr);
	ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB); CHKERRQ(ierr);
	ierr = MatView(A, viewer);	CHKERRQ(ierr);
	ierr = PetscViewerPopFormat(viewer); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

	return ierr;
}

PetscErrorCode DMD::printVecPYTHON(std::string sFileName,
		std::string sVectorName, Vec V) const {
	PetscErrorCode ierr;

	std::string sName = sFileName + ".csv";
	PetscViewer viewer;
	ierr = PetscObjectSetName((PetscObject) V, sVectorName.c_str()); CHKERRQ(ierr);
	ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, sName.c_str(), &viewer); CHKERRQ(ierr);
	ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_PYTHON); CHKERRQ(ierr);
	ierr = VecView(V, viewer);	CHKERRQ(ierr);
	ierr = PetscViewerPopFormat(viewer); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

	return ierr;
}

PetscErrorCode DMD::printMatPYTHON(std::string sFilename,
		std::string sMatrixName, Mat A) const {
	PetscErrorCode ierr;

	std::string sName = sFilename + ".csv";
	PetscViewer viewer;
	ierr = PetscObjectSetName((PetscObject) A, sMatrixName.c_str()); CHKERRQ(ierr);
	ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, sName.c_str(), &viewer); CHKERRQ(ierr);
	ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_PYTHON); CHKERRQ(ierr);
	ierr = MatView(A, viewer);	CHKERRQ(ierr);
	ierr = PetscViewerPopFormat(viewer); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

	return ierr;
}

void DMD::recordTime(std::chrono::steady_clock::time_point start,
		std::string sMessage){
#ifdef TIMING
	auto stop =	std::chrono::steady_clock::now();
	std::chrono::duration<double> duration = stop - start;
	fprintf(fLog, "%s: %f [s]\n", sMessage.c_str(), duration.count());
#endif
}


/*
 * This function does not consider complex numbers - SHOULD BE FIXED!!!
 */

PetscErrorCode DMD::calcDMDmodes(){
	PetscErrorCode ierr {};
	Eigen::MatrixXd eX2, eVr, eAtilde, eSr_inv, eSr;

	eX2 = pMat_to_eMat_double(X2);
	eVr = pMat_to_eMat_double(lrSVD.Vr);
	eAtilde = pMat_to_eMat_double(Atilde);
	eSr_inv = pMat_to_eMat_double(lrSVD.Sr_inv);
	eSr = pMat_to_eMat_double(lrSVD.Sr);

#ifdef ML
	Eigen::MatrixXcd mDMD_energy = lrSVD.eigVecs_small.inverse() * eSr * eVr;
	Eigen::ArrayXd DMD_energy_norms = mDMD_energy.real().rowwise().norm();
	Eigen::IOFormat CSVFmt(Eigen::FullPrecision, Eigen::DontAlignCols, ", ");
	if (fML.is_open()) {
		fML << DMD_energy_norms.transpose().format(CSVFmt) << ", ";
	}
#endif

//	Eigen::EigenSolver<Eigen::MatrixXd> eigenSystem;
//	eigenSystem.compute(eAtilde);
//	Eigen::VectorXcd eivals = eigenSystem.eigenvalues();
//	Eigen::MatrixXcd eW = eigenSystem.eigenvectors();

#ifdef TEST_EIGEN_SYSTEMS
	std::printf("\nTesting the result of eigen solver (Eigen3)...\n");
	std::printf("\tTesting if Aw = wL\n");
	for (int i = 0; i < eivals.size(); i++) {
		double epsError =  (eAtilde * eW.col(i) - eW.col(i) * eivals(i)).norm();
		if (epsError > 1e-16){
			std::cout << i << " eigenvector error: "<< epsError << std::endl;
		}
	}
#endif

	// Computing the DMD modes
	// These two are required in dot product functions below. So don't delete them yet
	epsPhi = eX2 * eVr * eSr_inv * lrSVD.eigVecs_small;
//	eigenPhi = eX2 * eVr * eSr_inv * eW;

	{
//		std::ofstream fPhi("Phi_DMD_"+std::to_string(isDMD_execs)+".dat");
//		if (fPhi.is_open()) {
//			fPhi << epsPhi.real() << '\n';
//		}
//		fPhi.close();

//		lrSVD.evOmega = eivals.array().log();
//		lrSVD.evOmega /= dt;
//		std::ofstream fOmega("Omega.dat");
//		if (fOmega.is_open()) {
//			fOmega << lrSVD.evOmega << '\n';
//		}
//		fOmega.close();

//		std::ofstream fEig("eigen3Eigs_DMD_"+std::to_string(isDMD_execs)+".dat");
//		if (fEig.is_open()) {
//			Eigen::MatrixXcd V_merged(eivals.rows(), 2);
//			V_merged << eivals, lrSVD.evOmega;
//			fEig << "Amp Factors\t\tEigenvalues\n";
//			fEig << std::left << V_merged;
//		}
//		fEig.close();
	}


#ifdef ML
	//---- Computing the time dynamics of the DMD modes -----//
	Eigen::MatrixXd eX1 = pMat_to_eMat_double(X1);
	Eigen::VectorXd V = eX1.col(0);

//	 Solving the least square problem
	Eigen::VectorXcd b = (epsPhi.transpose() * epsPhi).ldlt().solve(epsPhi.transpose() * V);

	//But I only need the time-dynamics at some time in far future
	//So don't need an array, and need only one far future time-step
	Eigen::ArrayXd time_dynamics = abs(b.array() * (lrSVD.omega_sorted.array()*1000).exp());

	if (fML.is_open()) {
		fML << time_dynamics.transpose().format(CSVFmt) << ", ";
	}
#endif

	return ierr;
}

PetscErrorCode DMD::dotwDMDmodes(const Vec& pVec, int numMode, bool bEPS){
	PetscErrorCode ierr = 0;
	Eigen::VectorXd eVec = pVec_to_eVec_double(pVec);

	assert(numMode <= svdRank &&
			"Requested mode to take dot product with, does not exist.\n");
	if (bEPS) {
	assert((eVec.size() == epsPhi.col(numMode).size()) && "the vectors are not the same size for taking dot product!\n\n");
	double dotMode = eVec.dot(epsPhi.col(numMode).real());
	std::printf("dot with EPS mode (sorted) %d:\t %f\n", numMode, dotMode);
	}

	// Becuase I am only keeping the ordered modes computed through SLEPc EPS
//	else {
//		assert((eVec.size() == eigenPhi.col(numMode).size()) && "the vectors are not the same size for taking dot product!\n\n");
//		double dotMode = eVec.dot(eigenPhi.col(numMode).real());
//		std::printf("dot with eigen3 mode (unsorted) %d:\t %f\n", numMode, dotMode);
//	}

	return ierr;
}


Eigen::MatrixXd pMat_to_eMat_double(const Mat &pMat){
	PetscInt rows{}, cols{};
	MatGetSize(pMat, &rows, &cols);
	std::vector<PetscInt> row_index, col_index;
	std::vector<double> mValues;

	for (int i = 0; i < rows; i++) {
		row_index.push_back(i);
	}
	for (int j = 0; j < cols; j++) {
		col_index.push_back(j);
		for (int i = 0; i < rows; i++) {
			mValues.push_back(0.0);
		}
	}

	MatGetValues(pMat, rows, row_index.data(), cols, col_index.data(), mValues.data());
	Eigen::Map<Eigen::MatrixXd> eMat(mValues.data(), rows, cols);

	return eMat;
}

Eigen::VectorXd pVec_to_eVec_double(const Vec &pVec){
	PetscInt nVals{};
	VecGetSize(pVec, &nVals);
	std::vector<PetscInt> index;
	std::vector<double> vValues;

	for (int i = 0; i < nVals; i++) {
		index.push_back(i);
		vValues.push_back(0.0);
	}

	VecGetValues(pVec, nVals, index.data(), vValues.data());
	Eigen::Map<Eigen::VectorXd> eVec(vValues.data(), nVals);

	return eVec;
}


// Deprecated
//PetscErrorCode DMD::computeUpdate(PetscInt iMode){
//	PetscErrorCode ierr;
//	PetscScalar TDend, slope, conv_sum;
//
//	assert(iMode < svdRank
//			&& "Requesting a mode that does not exist!! Did you set -dmd_rank correctly?");
//	slope = std::real(lrSVD.eigs[iMode]);
//
//	ierr = VecCreateSeq(PETSC_COMM_SELF, X.num_rows, &update); CHKERRQ(ierr);
//
//	ierr = MatGetValue(time_dynamics, X.num_cols - 1, iMode, &TDend); CHKERRQ(ierr);
//	conv_sum = slope * TDend / (1 - slope);
//
//	ierr = MatGetColumnVector(Phi, update, iMode); CHKERRQ(ierr);
//	ierr = VecScale(update, conv_sum); CHKERRQ(ierr);
//
//	return ierr;
//}

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

PetscErrorCode DMD::testEPS(){
	PetscErrorCode ierr = 0;
	Eigen::MatrixXd eAtilde;

	eAtilde = pMat_to_eMat_double(Atilde);

	std::cout << "EPS eigenvalues" << lrSVD.eigs << std::endl;

	std::printf("\nTesting the result of EPS (SLEPc)...\n");
	std::printf("\tTesting if Aw = wL\n");
	for (int i = 0; i < lrSVD.eigs.size(); i++) {
		double epsError =  (eAtilde * lrSVD.eigVecs_small.col(i) - lrSVD.eigVecs_small.col(i) * lrSVD.eigs(i)).norm();
		if (epsError > 1e-16){
			std::cout << i << " eigenvector error: "<< epsError << std::endl;
		}
	}
	return ierr;
}

PetscErrorCode DMD::testSVD(SVD& svd) {
	PetscErrorCode ierr;
	Vec u1, v1, u2, v2;
	PetscReal sigma1, sigma2;
	PetscScalar temp1, temp2;
	std::vector < PetscInt > index;
	std::vector<PetscScalar> column_vec, row_vec;
	Mat U, Vstar, SIGMA, USV;

	int cols{X.num_cols - 1}, rows{X.num_rows};
	PetscInt *VrCol_index;

	ierr = PetscMalloc1(cols, &VrCol_index);

	for (int i = 0; i < cols; i++){
		VrCol_index[i] = i;
	}

	ierr = MatCreateVecs(X1, &v1, &u1);
	CHKERRQ(ierr);
	ierr = MatCreateVecs(X1, &v2, &u2);
	CHKERRQ(ierr);
	index.push_back(0);
	for (int j = 0; j < cols; j++) {
		column_vec.push_back(0.0);
	}
	for (int i = 0; i < rows; i++) {
		row_vec.push_back(0.0);
	}

	ierr = MatCreate(MPI_COMM_SELF, &U);
	CHKERRQ(ierr);
	ierr = MatSetSizes(U, PETSC_DECIDE, PETSC_DECIDE, rows,
			cols);
	CHKERRQ(ierr);
	ierr = MatSetType(U, MATAIJ); // I am not sure about the matrix type rn.
	CHKERRQ(ierr);
	ierr = MatSetUp(U);
	CHKERRQ(ierr);

	ierr = MatCreate(MPI_COMM_SELF, &Vstar);
	CHKERRQ(ierr);
	ierr = MatSetSizes(Vstar, PETSC_DECIDE, PETSC_DECIDE, cols,
			cols);
	CHKERRQ(ierr);
	ierr = MatSetType(Vstar, MATAIJ); // I am not sure about the matrix type rn.
	CHKERRQ(ierr);
	ierr = MatSetUp(Vstar);
	CHKERRQ(ierr);

	ierr = MatCreate(MPI_COMM_SELF, &SIGMA);
	CHKERRQ(ierr);
	ierr = MatSetSizes(SIGMA, PETSC_DECIDE, PETSC_DECIDE, cols,
			cols);
	CHKERRQ(ierr);
	ierr = MatSetType(SIGMA, MATAIJ); // I am not sure about the matrix type rn.
	CHKERRQ(ierr);
	ierr = MatSetUp(SIGMA);
	CHKERRQ(ierr);

	ierr = MatZeroEntries(SIGMA);
	CHKERRQ(ierr);
	for (int j = 0; j < cols; j++) {
		index[0] = j;

		ierr = SVDGetSingularTriplet(svd, j, &sigma1, u1, v1);CHKERRQ(ierr);

		ierr = VecGetValues(u1, rows, row_index,row_vec.data());CHKERRQ(ierr);
		ierr = VecGetValues(v1, cols, VrCol_index,column_vec.data());CHKERRQ(ierr);
		ierr = MatSetValue(SIGMA, j, j, sigma1, INSERT_VALUES);	CHKERRQ(ierr);

		ierr = MatSetValues(U, rows, row_index, 1,index.data(), row_vec.data(), INSERT_VALUES);CHKERRQ(ierr);
		ierr = MatSetValues(Vstar, 1, index.data(), cols,VrCol_index, column_vec.data(), INSERT_VALUES);CHKERRQ(ierr);
	}

	ierr = MatAssemblyBegin(U, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);
	ierr = MatAssemblyEnd(U, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);
	ierr = MatAssemblyBegin(Vstar, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);
	ierr = MatAssemblyEnd(Vstar, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);
	ierr = MatAssemblyBegin(SIGMA, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);
	ierr = MatAssemblyEnd(SIGMA, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);

// SVD is performed on {A} to get {A=U \times \Sigma \times V^*}
// Refer to SLEPc documentation for more information
	std::printf("\nTesting the result of SVD...\n");

// U and V are unitary matrices {U^* \times U=I} and {V^* \times V=I}
	std::printf("\tTesting if U and V are unitary...\n");
	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < cols; j++) {
			ierr = SVDGetSingularTriplet(svd, i, &sigma1, u1, v1);
			CHKERRQ(ierr);

			ierr = SVDGetSingularTriplet(svd, j, &sigma2, u2, v2);
			CHKERRQ(ierr);

			ierr = VecDot(u1, u2, &temp1);
			CHKERRQ(ierr);

			ierr = VecDot(v1, v2, &temp2);
			CHKERRQ(ierr);

			if (i == j) {
				if (std::abs(temp1 - 1.0) > 1e-10)
					std::printf("\t{u[%i] \\dot u[%i]} is not equal to 1.0\n",
							i, j);
				if (std::abs(temp2 - 1.0) > 1e-10)
					std::printf("\t{v[%i] \\dot v[%i]} is not equal to 1.0\n",
							i, j);
			} else {
				if (std::abs(temp1 - 0.0) > 1e-10)
					std::printf("\t{u[%i] \\dot u[%i]} is not equal to 0.0\n",
							i, j);
				if (std::abs(temp2 - 0.0) > 1e-10)
					std::printf("\t{v[%i] \\dot v[%i]} is not equal to 0.0\n",
							i, j);
			}
		}
	}

// {A \times V = U \times \Sigma}
	std::printf("\tTesting if {A = U \\times \\Sigma \\times V^*}...\n");

	ierr = MatMatMatMult(U, SIGMA, Vstar, MAT_INITIAL_MATRIX, PETSC_DEFAULT,
			&USV);
	CHKERRQ(ierr);

	for (int j = 0; j < cols; j++) {
		for (int i = 0; i < rows; i++) {
			ierr = MatGetValue(X1, i, j, &temp1);
			CHKERRQ(ierr);

			ierr = MatGetValue(USV, i, j, &temp2);
			CHKERRQ(ierr);
			if (std::abs(temp1 - temp2) > 1e-10)
				std::printf(
						"\tThe result of SVD with index [%i][%i] is not equal to the same entry in the original matrix by %g!\n",
						i, j, std::abs(temp1 - temp2));
		}
	}

	return ierr;
}


