#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
	int m, n;
	double ** v;
} mat_t, *mat;

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
	for (int i = 0; i < x->m; i++)
		for (int j = 0; j < y->n; j++)
			for (int k = 0; k < x->n; k++)
				r->v[i][j] += x->v[i][k] * y->v[k][j];
	return r;
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
	mat x = matrix_new(n, n);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			x->v[i][j] = -2 *  v[i] * v[j];
	for (int i = 0; i < n; i++)
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
	mat q[m->m];
	mat z = m, z1;
	for (int k = 0; k < m->n && k < m->m - 1; k++) {
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
		z1 = matrix_mul(q[k], z);
		if (z != m) matrix_delete(z);
		z = z1;
	}
	matrix_delete(z);
	*Q = q[0];
	*R = matrix_mul(q[0], m);
	for (int i = 1; i < m->n && i < m->m - 1; i++) {
		z1 = matrix_mul(q[i], *Q);
		if (i > 1) matrix_delete(*Q);
		*Q = z1;
		matrix_delete(q[i]);
	}
	matrix_delete(q[0]);
	z = matrix_mul(*Q, m);
	matrix_delete(*R);
	*R = z;
	matrix_transpose(*Q);
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

int main()
{
	mat R, Q;
    int n_rows, n_cols;
    int i, j;
    mat x;
    mat y;
    mat Qt_y;
    mat coefs;
    mat y_pred;
    double mae;

    FILE *fin_features = fopen("insurance_features.txt", "r");
    fscanf(fin_features, "%d %d", &n_rows, &n_cols);

    x = matrix_new(n_rows, n_cols);
    for (i = 0; i < x->m; i++)
		for (j = 0; j < x->n; j++)
			fscanf(fin_features, "%lf", &x->v[i][j]);

    FILE *fin_target = fopen("insurance_target.txt", "r");
    y = matrix_new(n_rows, 1);
    for (i = 0; i < y->m; i++)
        fscanf(fin_target, "%lf", &y->v[i][0]);

	householder(x, &R, &Q);

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
	return 0;
}