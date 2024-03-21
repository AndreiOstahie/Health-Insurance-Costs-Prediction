build: par_omp par_mpi par_hybrid

par_omp: par_omp.o
	gcc par_omp.o -o par_omp -fopenmp -lm

par_omp.o: par_omp.c
	gcc -c par_omp.c -fopenmp

par_mpi: par_mpi.o
	mpicc par_mpi.o -o par_mpi -lm

par_mpi.o: par_mpi.c
	mpicc -c par_mpi.c

par_hybrid: par_hybrid.o
	mpicc par_hybrid.o -o par_hybrid -lm -fopenmp

par_hybrid.o: par_hybrid.c
	mpicc -fopenmp -c par_hybrid.c

run_omp:
	./par_omp

run_mpi:
	mpirun -np 4 --oversubscribe --mca btl ^openib ./par_mpi

run_hybrid:
	mpirun -np 4 --oversubscribe --mca btl ^openib ./par_hybrid

clean:
	rm -f par_omp par_mpi par_hybrid *.o