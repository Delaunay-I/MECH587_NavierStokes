CC = mpic++

# include ${PETSC_DIR}/conf/variables
# include ${PETSC_DIR}/conf/rules
# include /home/mirshahi/ANSLib/config/default-rules
# include /home/mirshahi/ANSLib/ANSLib-Defs.inc
# PETSC_ARCH_DIR=/usr/anslab/src/petsc/arch-linux2-cxx-debug
# include ${PETSC_ARCH_DIR}/lib/petsc/conf/petscvariables

# PETSC_INCLUDES=-I/usr/anslab/src/petsc/include -I${PETSC_ARCH_DIR}/include -I/home/mirshahi/ANSLib/src/adic-runtime


# Compiler flags
CFLAGS = -g #-v #-Wall
LINKER_FLAGS = -Wl,–verbose

# The build Target
CXX_FILES = NavierStokes.cpp allocation_IO.cpp validation_functions.cpp tri_test.cpp DMD.cxx
EXE_FILES = NavierStokes.o

EXTRA_FLAGS = -lm -lblas -llapack -llapacke 
INCLUDES = -I/$(SLEPC_DIR)/include -I/${SLEPC_DIR}/${PETSC_ARCH}/include \
-I/${PETSC_DIR}/include -I/${PETSC_DIR}/${PETSC_ARCH}/include 

LIBS = -L/$(SLEPC_DIR)/lib  \
-L/usr/lib \
-Wl,-rpath=/$(PETSC_DIR)/${PETSC_ARCH}/lib \
-Wl,-rpath=${SLEPC_DIR}/${PETSC_ARCH}/lib \
-L${SLEPC_DIR}/${PETSC_ARCH}/lib -lslepc \
-L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc 


#$(CXX_FILES)

# NavierStokes: ${EXE_FILES}
# 	@echo Making NavierStokes from $^
# 	${CC} ${CXX_FILES} \
# 	-o $@ \
# 	$(CFLAGS) ${EXTRA_FLAGS} \
# 	$(INCLUDES) $(LIBS) ${SLEPC} \

#	$(CC) -o $@ $(CFLAGS) ${EXTRA_FLAGS} $^ $(INCLUDES) $(LIBS) ${SLEPC}  
COMPILE.c = ${CC} ${CFLAGS}

OBJS = NavierStokes.o allocation_IO.o validation_functions.o tri_test.o DMD.o
main: ${OBJS}
	@echo Making $@ from $^
	@${COMPILE.c} $^ -o $@.o ${INCLUDES} ${LIBS} ${EXTRA_FLAGS}

NavierStokes.o: NavierStokes.cpp functions.h DMD.h
	@echo Making $@ from $^
	@${COMPILE.c} -c $< -o $@ ${INCLUDES} ${LIBS} ${EXTRA_FLAGS}

DMD.o: DMD.cxx DMD.h
	@${COMPILE.c} -c $< -o $@ ${INCLUDES}  ${LIBS} ${EXTRA_FLAGS}

allocation_IO.o: allocation_IO.cpp functions.h
	gcc -c $< -o $@

validation_functions.o: validation_functions.cpp functions.h
	gcc -c $< -o $@

tri_test.o: tri_test.cpp
	gcc -c $< -o $@



# NavierStokes.o: NavierStokes.cpp functions.h allocation_IO.cpp
# 	$(CC) $(CFLAGS) -c NavierStokes.cpp

# allocation_IO.o: functions.h NavierStokes.cpp
# 	@echo building $^

# validation_functions.o: functions.h

# tri_test.o:
# 	$(CC) $(CFLAGS) -c tri_test.cpp

# $(TARGET): $(CXX_FILES)
# 	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).cpp


clean:
	@echo Cleaning files...
	@$(RM) *.o