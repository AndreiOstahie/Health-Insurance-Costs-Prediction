#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define MASTER 0

typedef struct {
	int m, n;
	double ** v;
} mat_t, *mat;

int n_procs;
int rank;
int NUM_THREADS;

mat matrix_new(int m, int n)
{
	mat x = malloc(sizeof(mat_t));
	x->v = malloc(sizeof(double*) * m);
	x->v[0] = calloc(sizeof(double), m * n);
	for (int i = 0; i < m; i++)
		x->v[i] = x->v[0] + n * i;
	x->m = m;
	x->n = n;
	return x;
}

void matrix_delete(mat m)
{
	free(m->v[0]);
	free(m->v);
	free(m);
}

void matrix_transpose(mat m)
{
	for (int i = 0; i < m->m; i++) {
		for (int j = 0; j < i; j++) {
			double t = m->v[i][j];
			m->v[i][j] = m->v[j][i];
			m->v[j][i] = t;
		}
	}
}

mat matrix_copy(int n, double a[][n], int m)
{
	mat x = matrix_new(m, n);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			x->v[i][j] = a[i][j];
	return x;
}

mat matrix_mul(mat x, mat y)
{
	if (x->n != y->m) return 0;
	mat r = matrix_new(x->m, y->n);

    int i, j, k;

	for (i = 0; i < x->m; i++)
		for (j = 0; j < y->n; j++)
			for (k = 0; k < x->n; k++)
				r->v[i][j] += x->v[i][k] * y->v[k][j];
	return r;
}

mat matrix_mul_master(mat x, mat y)
{
	MPI_Status status;

	if (x->n != y->m) return 0;
	mat r = matrix_new(x->m, y->n);

	int rows_per_proc = x->m / n_procs;
	int remainder = x->m % n_procs;
	int rows = rows_per_proc;
	int master_rows = rows;

	if (remainder > 0)
		master_rows = rows_per_proc + 1;
	
	int offset = master_rows;

	for (int dest = 1; dest < n_procs; dest++) {
		if (dest < remainder)
			rows = rows_per_proc + 1;
		else
			rows = rows_per_proc;

		MPI_Send(&offset, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
		MPI_Send(&rows, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
		MPI_Send(&x->n, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
		MPI_Send(&y->n, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
		MPI_Send(&x->v[offset][0], rows * x->n, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
		MPI_Send(&y->v[0][0], x->n * y->n, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);

		offset = offset + rows;
	}

	int i, j, k;
    #pragma omp parallel for shared(r, x, y) private(i, j)
	for (k = 0; k < y->n; k++) {
		for (i = 0; i < master_rows; i++) {
			r->v[i][k] = 0.0;

			for (j = 0; j < x->n; j++) {
				r->v[i][k] += x->v[i][j] * y->v[j][k];
			}
		}
	}

	for (int src = 1; src < n_procs; src++) {
		MPI_Recv(&offset, 1, MPI_INT, src, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(&rows, 1, MPI_INT, src, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(&r->v[offset][0], rows * y->n, MPI_DOUBLE, src, 1, 
				MPI_COMM_WORLD, &status);
	}

	return r;
}

void matrix_mul_slave()
{
	MPI_Status status;
	int offset;
	int rows;
	int cols_a;
	int cols_b;

	MPI_Recv(&offset, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
	MPI_Recv(&rows, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
	MPI_Recv(&cols_a, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
	MPI_Recv(&cols_b, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);

	mat a = matrix_new(rows, cols_a);
	mat b = matrix_new(cols_a, cols_b);
	mat c = matrix_new(rows, cols_b);

	MPI_Recv(&a->v[0][0], rows * cols_a, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, &status);
	MPI_Recv(&b->v[0][0], cols_a * cols_b, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, &status);

	int i, j, k;
    #pragma omp parallel for shared(a, b, c) private(i, j)
	for (k = 0; k < cols_b; k++) {
		for (i = 0; i < rows; i++) {
			c->v[i][k] = 0.0;

			for (j = 0; j < cols_a; j++) {
				c->v[i][k] += a->v[i][j] * b->v[j][k];
			}
		}
	}

	MPI_Send(&offset, 1, MPI_INT, MASTER, 1, MPI_COMM_WORLD);
	MPI_Send(&rows, 1, MPI_INT, MASTER, 1, MPI_COMM_WORLD);
	MPI_Send(&c->v[0][0], rows * cols_b, MPI_DOUBLE, MASTER, 1, MPI_COMM_WORLD);
}

mat matrix_minor(mat x, int d)
{
	mat m = matrix_new(x->m, x->n);
	for (int i = 0; i < d; i++)
		m->v[i][i] = 1;
	for (int i = d; i < x->m; i++)
		for (int j = d; j < x->n; j++)
			m->v[i][j] = x->v[i][j];
	return m;
}

/* c = a + b * s */
double *vmadd(double a[], double b[], double s, double c[], int n)
{
	for (int i = 0; i < n; i++)
		c[i] = a[i] + s * b[i];
	return c;
}

/* m = I - v v^T */
mat vmul(double v[], int n)
{
    int i, j;
	mat x = matrix_new(n, n);

	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			x->v[i][j] = -2 *  v[i] * v[j];

	for (i = 0; i < n; i++)
		x->v[i][i] += 1;

	return x;
}

/* ||x|| */
double vnorm(double x[], int n)
{
	double sum = 0;
	for (int i = 0; i < n; i++) sum += x[i] * x[i];
	return sqrt(sum);
}

/* y = x / d */
double* vdiv(double x[], double d, double y[], int n)
{
	for (int i = 0; i < n; i++) y[i] = x[i] / d;
	return y;
}

/* take c-th column of m, put in v */
double* mcol(mat m, double *v, int c)
{
	for (int i = 0; i < m->m; i++)
		v[i] = m->v[i][c];
	return v;
}

void matrix_show(mat m)
{
	for(int i = 0; i < m->m; i++) {
		for (int j = 0; j < m->n; j++) {
			printf(" %8.3f", m->v[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

void householder(mat m, mat *R, mat *Q)
{
	mat *q;
	mat z, z1;

	int x_rows;
	int x_cols;

	if (rank == MASTER) {
		q = malloc(m->m * sizeof(mat));
		z = m;
		x_rows = m->m;
		x_cols = m->n;
	}

	MPI_Bcast(&x_rows, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&x_cols, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

	for (int k = 0; k < x_cols && k < x_rows - 1; k++) {
		MPI_Barrier(MPI_COMM_WORLD);

		if (rank == MASTER) {
			double e[m->m], x[m->m], a;
			z1 = matrix_minor(z, k);
			if (z != m) matrix_delete(z);
			z = z1;

			mcol(z, x, k);
			a = vnorm(x, m->m);
			if (m->v[k][k] > 0) a = -a;

			for (int i = 0; i < m->m; i++)
				e[i] = (i == k) ? 1 : 0;

			vmadd(x, e, a, e, m->m);
			vdiv(e, vnorm(e, m->m), e, m->m);
			q[k] = vmul(e, m->m);
		}

		if (rank == MASTER) {
			z1 = matrix_mul_master(q[k], z);
		}
		else {
			matrix_mul_slave();
		}

		if (rank == MASTER) {
			if (z != m) matrix_delete(z);
			z = z1;
		}
	}

	if (rank == MASTER) {
		matrix_delete(z);
		*Q = q[0];
	}
	
	if (rank == MASTER)
		*R = matrix_mul_master(q[0], m);
	else
		matrix_mul_slave();

	for (int i = 1; i < x_cols && i < x_rows - 1; i++) {
		if (rank == MASTER) {
			z1 = matrix_mul_master(q[i], *Q);
		} else {
			matrix_mul_slave();
		}

		if (rank == MASTER) {
			if (i > 1) matrix_delete(*Q);
			*Q = z1;
			matrix_delete(q[i]);
		}
	}

	if (rank == MASTER) {
		matrix_delete(q[0]);
	}

	if (rank == MASTER) {
		z = matrix_mul_master(*Q, m);
	} else {
		matrix_mul_slave();
	}
	
	if (rank == MASTER) {
		matrix_delete(*R);
		*R = z;
		matrix_transpose(*Q);

		free(q);
	}
}

mat upper_triangular_solve(mat X, mat y)
{
    int i, j;
    mat coefs = matrix_new(X->n, 1);

    for (i = X->n - 1; i >= 0; i--) {
        for (j = X->n - 1; j > i; j--) {
            y->v[i][0] -= X->v[i][j] * coefs->v[j][0];
        } 
        
        if (X->v[i][i] == 0) {
            printf("Division by 0 when solving upper triangular, at R[%d][%d]\n", i, i);
            exit(-1);
        }

        coefs->v[i][0] = y->v[i][0] / X->v[i][i];
    }

    return coefs;
}

mat predict(mat X, mat coefs)
{
    int i, j;
    mat y_pred = matrix_new(X->m, 1);

    for (i = 0; i < X->m; i++) {
        mat xi = matrix_new(1, X->n);
        mat yi;

        for (j = 0; j < X->n; j++) {
            xi->v[0][j] = X->v[i][j];
        }

        yi = matrix_mul(xi, coefs);
        y_pred->v[i][0] = yi->v[0][0];

        matrix_delete(xi);
        matrix_delete(yi);
    }

    return y_pred;
}

double mean_absolute_error(mat y_true, mat y_pred)
{
    int i;
    double sum;
    int m = y_true->m;

    for (i = 0; i < m; i++) {
        sum += abs(y_true->v[i][0] - y_pred->v[i][0]);
    }

    return sum / m;
}

int main(int argc, char *argv[])
{
    // Set number of threads for OMP
	NUM_THREADS = atoi(argv[1]);
    omp_set_num_threads(NUM_THREADS);

	MPI_Init(NULL, NULL);

	mat R, Q;
    int n_rows, n_cols;
    int i, j;
    mat x;
    mat y;
    mat Qt_y;
    mat coefs;
    mat y_pred;
    double mae;
	FILE *fin_features;
	FILE *fin_target;

    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == MASTER) {
		fin_features = fopen("insurance_features.txt", "r");
		fscanf(fin_features, "%d %d", &n_rows, &n_cols);

		x = matrix_new(n_rows, n_cols);
		for (i = 0; i < x->m; i++)
			for (j = 0; j < x->n; j++)
				fscanf(fin_features, "%lf", &x->v[i][j]);

		fin_target = fopen("insurance_target.txt", "r");
		y = matrix_new(n_rows, 1);
		for (i = 0; i < y->m; i++)
			fscanf(fin_target, "%lf", &y->v[i][0]);
	}

	householder(x, &R, &Q);
	
	if (rank == MASTER) {
		matrix_transpose(Q);
		Qt_y = matrix_mul(Q, y);
		coefs = upper_triangular_solve(R, Qt_y);

		puts("Coefs:"); matrix_show(coefs);

		y_pred = predict(x, coefs);
		mae = mean_absolute_error(y, y_pred);
		printf("Mean Absolute Error: %lf\n", mae);

		matrix_delete(x);
		matrix_delete(R);
		matrix_delete(Q);
		matrix_delete(y);
		matrix_delete(Qt_y);
		matrix_delete(coefs);
		matrix_delete(y_pred);

		fclose(fin_features);
		fclose(fin_target);
	}

	MPI_Finalize();
	return 0;
}