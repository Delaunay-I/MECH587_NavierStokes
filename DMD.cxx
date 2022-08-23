/*
 * DMD.cpp
 *
 *  Created on: Jul 30, 2022
 *      Author: mirshahi
 */

#include "DMD.h"

DMD::DMD(const Mat *svdMat, PetscInt iModes, PetscReal DT) :
		 iNumModes(iModes), dt(DT) {


	X.mat = svdMat;
	MatGetSize(*X.mat, &X.num_rows, &X.num_cols);
	PetscMalloc1(X.num_rows, &row_index);

	for (int i = 0; i < X.num_rows; i++) {
		row_index[i] = i;
	}

	PetscBool flg;

	PetscOptionsGetInt(NULL, PETSC_NULL, "-dmd_rank", &svdRank, &flg);

	if (!flg) {
		PetscErrorPrintf("Missing -dmd_rank flag!\n");
		exit(2);
	} else if (svdRank > X.num_cols) {
		PetscErrorPrintf("\nSVD rank is greater that number of svd matrices.\n "
				"Are you collecting setting up the SVD matrix correctly?\n");
		exit(2);
	}


}


DMD::~DMD() {
	PetscFree(row_index);
	MatDestroy(&X1);
	MatDestroy(&X2);
	MatDestroy(&Ur);
	MatDestroy(&Sr);
	MatDestroy(&Vr);
	MatDestroy(&Sr_inv);
	MatDestroy(&W);
	MatDestroy(&Atilde);
	MatDestroy(&Phi);
	MatDestroy(&time_dynamics);
	VecDestroy(&update);



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

	ierr = PetscPrintf(PETSC_COMM_WORLD,
			"Preparing matrices for Dynamic mode decomposition...\n");
	CHKERRQ(ierr);

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
	printMatPYTHON("X", "X", *X.mat);
	printMatMATLAB("X", "X", *X.mat);

	printMatPYTHON("X1", "X1", X1);
	printMatPYTHON("X2", "X2", X2);
#endif

	ierr = PetscFree(X1col_index); CHKERRQ(ierr);
	ierr = PetscFree(X2col_index); CHKERRQ(ierr);
	ierr = PetscFree(dTmpArr); CHKERRQ(ierr);

	return ierr;
}

PetscErrorCode DMD::solveSVD() {
	PetscErrorCode ierr;
	SVD svd;
	PetscInt nsv { svdRank };

	ierr = SVDCreate(PETSC_COMM_WORLD, &svd);
	CHKERRQ(ierr);
	ierr = SVDSetOperator(svd, X1); // This function is changed to SVDSetOperators(SVD svd,Mat A,Mat B) in the new version of SLEPC.
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

	/*
	 * Getting Singular values and singular vectors
	 * and building low rank truncation matrices
	 */


	Vec u, v;
	PetscReal sigma;
	PetscInt numCols{svdRank};
	PetscInt *VrCol_index;
	std::vector<PetscInt> index(1);
	PetscScalar *column_vec, *row_vec;

	ierr = MatCreateVecs(X1, &v, &u);CHKERRQ(ierr); //vector that the matrix vector product can be stored in

	ierr = PetscMalloc1(X.num_cols - 1, &column_vec); CHKERRQ(ierr);
	ierr = PetscMalloc1(X.num_rows, &row_vec); CHKERRQ(ierr);
	ierr = PetscMalloc1(X.num_cols - 1, &VrCol_index);

	for (int i = 0; i < X.num_cols - 1; i++){
		VrCol_index[i] = i;
	}


	ierr = MatDestroy(&Ur);	CHKERRQ(ierr);
	ierr = MatDestroy(&Sr);	CHKERRQ(ierr);
	ierr = MatDestroy(&Vr);	CHKERRQ(ierr);

	ierr = MatCreate(PETSC_COMM_WORLD, &Ur); CHKERRQ(ierr);
	ierr = MatCreate(PETSC_COMM_WORLD, &Sr); CHKERRQ(ierr);
	ierr = MatCreate(PETSC_COMM_WORLD, &Vr);CHKERRQ(ierr);

	ierr = MatSetSizes(Ur, PETSC_DECIDE, PETSC_DECIDE, X.num_rows, numCols);
	CHKERRQ(ierr);
	ierr = MatSetSizes(Sr, PETSC_DECIDE, PETSC_DECIDE, numCols, numCols);
	CHKERRQ(ierr);
	ierr = MatSetSizes(Vr, PETSC_DECIDE, PETSC_DECIDE, X.num_cols - 1, numCols);
	CHKERRQ(ierr);

	ierr = MatSetType(Ur, MATAIJ);
	CHKERRQ(ierr);
	ierr = MatSetType(Sr, MATAIJ); CHKERRQ(ierr);
	ierr = MatSetType(Vr, MATAIJ); CHKERRQ(ierr);

	ierr = MatSetUp(Ur);	CHKERRQ(ierr);
	ierr = MatSetUp(Sr);	CHKERRQ(ierr);
	ierr = MatSetUp(Vr);	CHKERRQ(ierr);

	ierr = MatZeroEntries(Ur); CHKERRQ(ierr);
	ierr = MatZeroEntries(Sr); CHKERRQ(ierr); //zeros all entries of a matrix
	ierr = MatZeroEntries(Vr); CHKERRQ(ierr);

	ierr = PetscPrintf(PETSC_COMM_WORLD,
			"Setting up singular values and solution modes...\n");
	CHKERRQ(ierr);

		/* Getting SVD modes column by column */
	for (int j = 0; j < numCols; j++) {
		index[0] = j;
		ierr = PetscPrintf(PETSC_COMM_WORLD,
					"Singular value number: %i \n", index[0]);
			CHKERRQ(ierr);

		ierr = SVDGetSingularTriplet(svd, j, &sigma, u, v); CHKERRQ(ierr);
		ierr = MatSetValue(Sr, j, j, sigma, INSERT_VALUES);	CHKERRQ(ierr);

		ierr = VecGetValues(u, X.num_rows, row_index, row_vec);
		CHKERRQ(ierr);
		ierr = MatSetValues(Ur, X.num_rows, row_index, 1, index.data(),
				row_vec, INSERT_VALUES);
		CHKERRQ(ierr);

		ierr = VecGetValues(v, X.num_cols - 1, VrCol_index, column_vec);
		CHKERRQ(ierr);
		ierr = MatSetValues(Vr, X.num_cols - 1, VrCol_index, 1, index.data(),
						column_vec, INSERT_VALUES);
		CHKERRQ(ierr);

	}

	ierr = MatAssemblyBegin(Ur, MAT_FINAL_ASSEMBLY);	CHKERRQ(ierr);
	ierr = MatAssemblyEnd(Ur, MAT_FINAL_ASSEMBLY);	CHKERRQ(ierr);
	ierr = MatAssemblyBegin(Sr, MAT_FINAL_ASSEMBLY);	CHKERRQ(ierr);
	ierr = MatAssemblyEnd(Sr, MAT_FINAL_ASSEMBLY);	CHKERRQ(ierr);
	ierr = MatAssemblyBegin(Vr, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(Vr, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);


#ifdef DMD_SIGMARATIO
	PetscScalar sigma1, sigma2;
	ierr = MatGetValue(Sr, 0, 0, &sigma1); CHKERRQ(ierr);
	ierr = MatGetValue(Sr, 1, 1, &sigma2); CHKERRQ(ierr);

	ierr = PetscPrintf(PETSC_COMM_WORLD, "\nSigma ratio is: %3f\n", sigma1/sigma2); CHKERRQ(ierr);
#endif

#ifdef DEBUG_DMD
	printMatPYTHON("Ur", "Ur", Ur);
	printMatMATLAB("Sr", "Sr", Sr);
	printMatPYTHON("Vr", "Vr", Vr);
#endif

	PetscFree(column_vec);
	PetscFree(row_vec);

	ierr = VecDestroy(&u); CHKERRQ(ierr);
	ierr = VecDestroy(&v); CHKERRQ(ierr);
	ierr = SVDDestroy(&svd); CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode DMD::regression() {
	PetscErrorCode ierr;
	Vec vDiagSr;
	Mat UrT_X2;
	PetscInt nconv;

	assert(svdRank >= 1
			&& "MATRIX INVERSION: SVD rank is less than one. Have you defined it?");
	// Invert Sr matrix - reciprocal of the diagonal
	ierr = VecCreateSeq(PETSC_COMM_SELF, svdRank, &vDiagSr); CHKERRQ(ierr);
	ierr = MatGetDiagonal(Sr, vDiagSr); CHKERRQ(ierr);
	ierr = VecReciprocal(vDiagSr); CHKERRQ(ierr);
	ierr = MatDuplicate(Sr, MAT_COPY_VALUES, &Sr_inv); CHKERRQ(ierr);
	ierr = MatDiagonalSet(Sr_inv, vDiagSr, INSERT_VALUES); CHKERRQ(ierr);

	printMatMATLAB("Sr_inv", "Sr_inv", Sr_inv);

	// Constructing Atilde
	ierr = MatTransposeMatMult(Ur, X2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &UrT_X2); CHKERRQ(ierr);
	ierr = MatMatMatMult(UrT_X2, Vr, Sr_inv, MAT_INITIAL_MATRIX, PETSC_DEFAULT,
					&Atilde); CHKERRQ(ierr);
#ifdef DEBUG_DMD
	printMatPYTHON("urtx2", "urtx2", UrT_X2);
	printMatPYTHON("Atilde", "Atilde", Atilde);
#endif

	/* ------------ Eigen analysis of Atilde -------------*/
	EPS eps;
	ierr = EPSCreate(PETSC_COMM_WORLD, &eps); CHKERRQ(ierr);

	ierr = EPSSetOperators(eps, Atilde, PETSC_NULL); CHKERRQ(ierr);
	ierr = EPSSetProblemType(eps, EPS_NHEP); CHKERRQ(ierr);
	ierr = EPSSetType(eps, EPSLAPACK); CHKERRQ(ierr);
	ierr = EPSSetDimensions(eps, svdRank, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
	ierr = EPSSetTolerances(eps, 1e-10, 10); CHKERRQ(ierr);
	ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);

	ierr = PetscPrintf(PETSC_COMM_WORLD,
			"\nSolving the eigensystem subspace...\n");
	CHKERRQ(ierr);
	ierr = EPSSolve(eps);	CHKERRQ(ierr);

	ierr = EPSGetConverged(eps, &nconv);
			CHKERRQ(ierr);
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

	/* ------ Extracting eigenpair --------- */

	Vec vr, vi;
	PetscScalar dVecr, dVeci, value;
	PetscScalar dReal, dImag, dError;

	ierr = MatCreateVecs(Atilde, NULL, &vr);
	CHKERRQ(ierr);
	ierr = MatCreateVecs(Atilde, NULL, &vi);
	CHKERRQ(ierr);

	ierr = MatDestroy(&W);
	CHKERRQ(ierr);
	ierr = MatCreate(PETSC_COMM_WORLD, &W);
	CHKERRQ(ierr);
	ierr = MatSetSizes(W, PETSC_DECIDE, PETSC_DECIDE, svdRank, svdRank);
	CHKERRQ(ierr);
	ierr = MatSetType(W, MATAIJ);
	CHKERRQ(ierr);
	ierr = MatSetUp(W);
	CHKERRQ(ierr);
	ierr = MatZeroEntries(W);
	CHKERRQ(ierr);

	ierr = PetscPrintf(PETSC_COMM_WORLD,
			"\nExtracting eigenvector and eigenvalues...\n");
	CHKERRQ(ierr);

	for (PetscInt i = 0; i < nconv; i++) {

		ierr = EPSGetEigenpair(eps, i, &dReal, &dImag, vr, vi);
		CHKERRQ(ierr);
		ierr = EPSComputeError(eps, i, EPS_ERROR_RELATIVE, &dError);
					CHKERRQ(ierr);
					eigs.push_back({dReal, dImag});
#ifdef PRINT_EIGENVALUES
			printf("eigen %i= Real: %f  imag: %f\n", i, std::real(eigs[i]), std::imag(eigs[i]));
#endif

		for (int row = 0; row < svdRank; row++) {
			//Get values one-by-one and write them one at a time
			ierr = VecGetValues(vr, 1, &row, &dVecr);
			CHKERRQ(ierr);
			ierr = VecGetValues(vi, 1, &row, &dVeci);
			CHKERRQ(ierr);
			// Setting the absolute value
			//value = std::sqrt(dVecr * dVecr + dVeci * dVeci);
			// Setting the real part
			value = dVecr; //real part only
			ierr = MatSetValue(W, row, i, value, INSERT_VALUES);
			CHKERRQ(ierr);

		}
	}
	ierr = MatAssemblyBegin(W, MAT_FINAL_ASSEMBLY);	CHKERRQ(ierr);
	ierr = MatAssemblyEnd(W, MAT_FINAL_ASSEMBLY);	CHKERRQ(ierr);

#ifdef DEBUG_DMD
	ierr = printMatPYTHON("W", "W", W); CHKERRQ(ierr);
	ierr = printMatMATLAB("W", "W", W); CHKERRQ(ierr);
#endif

	ierr = MatDestroy(&UrT_X2); CHKERRQ(ierr);
	ierr = VecDestroy(&vDiagSr); CHKERRQ(ierr);
	ierr = VecDestroy(&vr); CHKERRQ(ierr);
	ierr = VecDestroy(&vi); CHKERRQ(ierr);
	ierr = EPSDestroy(&eps); CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode DMD::calcDMDmodes(){
	PetscErrorCode ierr;
	Mat X2_Vr;

	// Calculating Spatial modes
		ierr = MatMatMult(X2, Vr, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &X2_Vr); CHKERRQ(ierr);
		ierr = MatMatMatMult(X2_Vr, Sr_inv, W, MAT_INITIAL_MATRIX, PETSC_DEFAULT,
						&Phi); CHKERRQ(ierr);

		std::vector<PetscReal> omega;
		for (size_t i = 0; i < eigs.size(); i++){
			PetscReal eig_real = std::real(eigs[i]);
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
	ierr = printVecMATLAB("x1", "x1", rhs); CHKERRQ(ierr);
	ierr = printVecMATLAB("b", "b", Soln); CHKERRQ(ierr);

#endif
	ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);

	ierr = KSPSetOperators(ksp, Phi, Phi); CHKERRQ(ierr);
	ierr = KSPSetType(ksp, KSPLSQR); CHKERRQ(ierr);
	ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
	ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);
	ierr = KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

	//ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

	ierr = KSPSolve(ksp, rhs, Soln); CHKERRQ(ierr);
	ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

#ifdef DEBUG_DMD
	/*
	 * The Solution of the least squares (b) is related to the eigenvectors and
	 * the fact that we are not including imaginery parts in the KSP solver
	 * So the results are diffferent from Python and MATLAB
	 */
	ierr = printVecMATLAB("b", "b", Soln); CHKERRQ(ierr);
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
	ierr = printMatPYTHON("TD", "TD", time_dynamics); CHKERRQ(ierr);
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
	ierr = printMatPYTHON("Ir", "Ir", Ir); CHKERRQ(ierr);
	ierr = printMatPYTHON("lhs_inv", "lhs_inv", lhs); CHKERRQ(ierr);
	ierr = printMatPYTHON("full", "full", Gtilde); CHKERRQ(ierr);
#endif

	Mat mTmp{}, mTmp2{};
	Vec X2_end{};
	ierr = VecCreateSeq(PETSC_COMM_SELF, X.num_rows, &X2_end); CHKERRQ(ierr);
	ierr = VecCreateSeq(PETSC_COMM_SELF, X.num_rows, &update); CHKERRQ(ierr);

	ierr = MatGetColumnVector(X2, X2_end, X.num_cols - 2); CHKERRQ(ierr);
	ierr = MatMatMult(Ur, Gtilde, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &mTmp); CHKERRQ(ierr);
	ierr = MatMatTransposeMult(mTmp, Ur, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &mTmp2); CHKERRQ(ierr);


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
	slope = std::real(eigs[iMode]);

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
	ierr = solveSVD();
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
	ierr = solveSVD();CHKERRQ(ierr);
	ierr = regression();CHKERRQ(ierr);
	ierr = calcDMDmodes();CHKERRQ(ierr);
	ierr = computeMatTransUpdate();CHKERRQ(ierr);

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



