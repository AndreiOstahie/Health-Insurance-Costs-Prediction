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
    int n_rows, n_cols;
	mat coefs;
    int i, j;
    mat x, y;
    mat y_pred;
    double mae;

    FILE *fin_features = fopen("insurance_features_test.txt", "r");
    fscanf(fin_features, "%d %d", &n_rows, &n_cols);

    x = matrix_new(n_rows, n_cols);
    for (i = 0; i < x->m; i++)
		for (j = 0; j < x->n; j++)
			fscanf(fin_features, "%lf", &x->v[i][j]);

    FILE *fin_target = fopen("insurance_target_test.txt", "r");
    y = matrix_new(n_rows, 1);
    for (i = 0; i < y->m; i++)
        fscanf(fin_target, "%lf", &y->v[i][0]);

    FILE *fin_coefs = fopen("saved_coefs.txt", "r");
    fscanf(fin_coefs, "%d %d", &n_rows, &n_cols);
    
    coefs = matrix_new(n_rows, n_cols);
    for (i = 0; i < coefs->m; i++)
		for (j = 0; j < coefs->n; j++)
			fscanf(fin_coefs, "%lf", &coefs->v[i][j]);

    puts("Loaded coefficients:"); matrix_show(coefs);

    y_pred = predict(x, coefs);
    mae = mean_absolute_error(y, y_pred);
    printf("Test Mean Absolute Error: %lf\n", mae);

	matrix_delete(x);
    matrix_delete(y);
    matrix_delete(coefs);
    matrix_delete(y_pred);

    fclose(fin_features);
    fclose(fin_target);
    fclose(fin_coefs);
	return 0;
}