CC = mpic++

# Compiler flags
CFLAGS = -g #-v #-Wall
LINKER_FLAGS = -Wl,–verbose

EXTRA_FLAGS = -lm -lblas -llapack -llapacke -lpetsc -lslepc
INCLUDES = -I/$(SLEPC_DIR)/include -I/${SLEPC_DIR}/${PETSC_ARCH}/include \
-I/${PETSC_DIR}/include -I/${PETSC_DIR}/${PETSC_ARCH}/include 

LIBS = -L/$(SLEPC_DIR)/lib  \
-L/usr/lib \
-Wl,-rpath=/$(PETSC_DIR)/${PETSC_ARCH}/lib \
-Wl,-rpath=${SLEPC_DIR}/${PETSC_ARCH}/lib \
-L${SLEPC_DIR}/${PETSC_ARCH}/lib  \
-L${PETSC_DIR}/${PETSC_ARCH}/lib  

COMPILE.c = ${CC} ${CFLAGS}

OBJS = NavierStokes.o allocation_IO.o validation_functions.o tri_test.o DMD.o
main: ${OBJS}
	@echo Building $@ from $^
	@${COMPILE.c} $^ -o $@.o ${INCLUDES} ${LIBS} ${EXTRA_FLAGS}

NavierStokes.o: NavierStokes.cpp functions.h DMD.h
	@echo Building $@ from $^
	@${COMPILE.c} -c $< -o $@ ${INCLUDES} ${LIBS} ${EXTRA_FLAGS}

DMD.o: DMD.cxx DMD.h
	@echo Building $@ from $^
	@${COMPILE.c} -c $< -o $@ ${INCLUDES}  ${LIBS} ${EXTRA_FLAGS}

allocation_IO.o: allocation_IO.cpp functions.h
	@echo Building $@ from $^
	@gcc -c $< -o $@

validation_functions.o: validation_functions.cpp functions.h
	@echo Building $@ from $^
	@gcc -c $< -o $@

tri_test.o: tri_test.cpp
	@echo Building $@ from $^
	@gcc -c $< -o $@

clean:
	@echo Cleaning files...
	@$(RM) *.o


CXX_FILES = NavierStokes.cpp allocation_IO.cpp validation_functions.cpp tri_test.cpp DMD.cxx
EXE_FILES = NavierStokes.o