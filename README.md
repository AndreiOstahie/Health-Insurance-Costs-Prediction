# Health Insurance Costs Prediction #

### Descriere ###
Proiectul consta in efectuarea unei regresii liniare pentru estimarea costurilor medicale facturate
de asigurarea de sanatate pentru o persoana, folosind metoda de descompunere QR a unei matrice.


### Justificare ###
Ne propunem sa prezicem costurile asociate asigurarii medicale pentru o persoana, avand la dispozitie diferite caracteristici ale acesteia: varsta, regiune, fumator/nefumator etc. Modelul de regresie liniara multivariata este potrivit in acest caz.

Regresia liniara este folosita in statistica pentru a modela o relatie liniara dintre o variabila
dependenta si una sau mai multe variabile independente.
Practic, consta in determinarea coeficientilor b=[b0, b1, ..., bn] din ecuatia:

```
y = b0 + b1*x1 + b2*x2 + ... + bn * xn
```

Descompunerea QR reprezinta o metoda folosita in algebra liniara pentru a descompune o matrice
intr-un produs dintre o matrice ortogonala (Q) si una superior triunghiulara (R).

Folosind descompunerea QR, coeficientii b se pot afla usor din sistemul de ecuatii:

```
R*b = Q_T * y
```

Pentru o matrice X cu m linii si n coloane, complexitatea descompunerii QR este O(min(m, n) * m * n^2).

Desi metode din invatarea automata, precum Gradient Descent scaleaza mai bine la un set de date mai mare, metode bazate pe abordari din algebra liniara au o stabilitate numerica mai buna si functioneaza bine pe seturi de date mai mici. Metodele din algebra liniara nu necesita alegerea diferitilor hiperparametri, precum rata de invatare (learning rate).


### Plan ###
Paralelizarea algoritmului se va face folosind OpenMP si MPI.

In etapa M2 vom aborda paralelizarea folosind OpenMP (implementare + calcul speedup).

In etapa M3 vom aborda paralelizarea folosind MPI (implementare + calcul speedup).

In etapa M4 vom face o analiza detaliata pentru performanta paralela a implementarilor. De asemenea, vom actualiza README-ul cu noi date utile.

### Etapa M0 si M1 ###

Am obtinut dataset-ul cu costurile de asigurare de sanatate pentru diferiti indivizi si l-am preprocesat
pentru a putea fi folosit usor in implementari.

Am implementat varianta seriala si am facut profiling folosind gprof:

| %     | cumulative | self    | self  | total  |        |                        |
|-------|------------|---------|-------|--------|--------|------------------------|
| time  | seconds    | seconds | calls | s/call | s/call | name                   |
| 99.94 | 61.89      | 61.89   | 1018  | 0.06   | 0.06   | matrix_mul             |
| 0.06  | 61.93      | 0.04    | 8     | 0.01   | 0.01   | vmul                   |
| 0.00  | 61.93      | 0.00    | 2038  | 0.00   | 0.00   | matrix_delete          |
| 0.00  | 61.93      | 0.00    | 2038  | 0.00   | 0.00   | matrix_new             |
| 0.00  | 61.93      | 0.00    | 16    | 0.00   | 0.00   | vnorm                  |
| 0.00  | 61.93      | 0.00    | 8     | 0.00   | 0.00   | matrix_minor           |
| 0.00  | 61.93      | 0.00    | 8     | 0.00   | 0.00   | mcol                   |
| 0.00  | 61.93      | 0.00    | 8     | 0.00   | 0.00   | vdiv                   |
| 0.00  | 61.93      | 0.00    | 8     | 0.00   | 0.00   | vmadd                  |
| 0.00  | 61.93      | 0.00    | 2     | 0.00   | 0.00   | matrix_transpose       |
| 0.00  | 61.93      | 0.00    | 1     | 0.00   | 1.07   | householder            |
| 0.00  | 61.93      | 0.00    | 1     | 0.00   | 0.00   | matrix_show            |
| 0.00  | 61.93      | 0.00    | 1     | 0.00   | 0.00   | mean_absolute_error    |
| 0.00  | 61.93      | 0.00    | 1     | 0.00   | 60.80  | predict                |
| 0.00  | 61.93      | 0.00    | 1     | 0.00   | 0.00   | upper_triangular_solve |

Am creat repository-ul pe Gitlab, am definit scopul proiectului si planul pentru urmatoarele etape.

### Etapa M2 ###

Am implementat paralelizarea algoritmului folosind OpenMP, si am obtinut timpii de rulare pentru un numar de thread-uri intre 2 si 24.

Graficul timpului de executie in functie de numarul de thread-uri:

![Timp de executie - OpenMP](performance/omp/omp_execution_time.png?raw=true "Timp de executie - OpenMP")

Speedup-ul obtinut in functie de numarul de thread-uri:

![Speedup - OpenMP](performance/omp/omp_speedup.png?raw=true "Speedup - OpenMP")

### Etapa M3 ###

Am implementat paralelizarea algoritmului folosind MPI, si am obtinut timpii de rulare pentru un numar de procese intre 2 si 24.

Graficul timpului de executie in functie de numarul de procese:

![Timp de executie - MPI](performance/mpi/mpi_execution_time.png?raw=true "Timp de executie - MPI")

Speedup-ul obtinut in functie de numarul de procese:

![Speedup - MPI](performance/mpi/mpi_speedup.png?raw=true "Speedup - MPI")

### Etapa M4 ###

Am analizat performanta implementarilor folosind Intel VTune Profiler.

Serial:

![Hotspots - serial](profiling/vtune/serial_hotspots.png?raw=true "Hotspots - serial")

OpenMP:

![Hotspots - OpenMP](profiling/vtune/omp_hotspots.png?raw=true "Hotspots - OpenMP")

![Microarchitecture - OpenMP](profiling/vtune/omp_microarchitecture.png?raw=true "Microarchitecture - OpenMP")

MPI:

![Hotspots - MPI](profiling/vtune/mpi_hotspots.png?raw=true "Hotspots - MPI")

![Bottom-up Call Stack - MPI](profiling/vtune/mpi_time_on_cpu.png?raw=true "Bottom-up Call Stack - MPI")

### Etapa M5 ###

Am implementat o varianta hibrida MPI + OpenMP.

Am validat rezultatele si am verificat corectitudinea lor folosind un script de testare.

Am pregatit un demo pentru prezentare.
