#!/bin/bash

mkdir out
serial_out="out/serial"
omp_out="out/omp"
mpi_out="out/mpi"
hybrid_out="out/hybrid"

serial_exec="./serial"
omp_exec="./par_omp"
mpi_exec="./par_mpi"
hybrid_exec="./par_hybrid"

max_threads=24
max_procs=24
n_runs=2


echo "Building ..."
make
cd ..
make
cd parallel
cp ../serial .

echo "Running serial program..."
$serial_exec >> $serial_out


echo "Testing OMP program..."
for ((i = 2 ; i <= $max_threads ; i+=2)); do
	echo "==================== $i THREADS ===================="
	for ((j = 0; j < $n_runs; j++)); do
		$omp_exec $i >> $omp_out
		if diff $omp_out $serial_out > /dev/null; then 
			echo "Run $j..........CORRECT";
		else
			echo "Run $j..........WRONG";
		fi
		rm $omp_out
	done
done

echo "Testing MPI program..."
for ((i = 2 ; i <= $max_procs ; i+=2)); do
	echo "==================== $i PROCESSES ===================="
	for ((j = 0; j < $n_runs; j++)); do
		mpirun -np $i --oversubscribe --mca btl ^openib $mpi_exec >> $mpi_out
		if diff $mpi_out $serial_out > /dev/null; then 
			echo "Run $j..........CORRECT";
		else
			echo "Run $j..........WRONG";
		fi
		rm $mpi_out
	done
done

echo "Testing hybrid program..."
for ((i = 2 ; i <= 4 ; i++)); do
	for ((j = 2 ; j <= 4 ; j++)); do
		echo "==================== $i PROCESSES $j THREADS ===================="
		for ((k = 0; k < $n_runs; k++)); do
			mpirun -np $i --oversubscribe --mca btl ^openib $hybrid_exec $j >> $hybrid_out
			if diff $hybrid_out $serial_out > /dev/null; then 
				echo "Run $k..........CORRECT";
			else
				echo "Run $k..........WRONG";
			fi
			rm $hybrid_out
		done
	done
	
done


echo "Cleaning..."
make clean
rm serial
rm -r out
cd ..
make clean
echo "Done."
