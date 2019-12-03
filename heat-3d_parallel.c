#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>


#define MIN(x,y) (x < y ? x : y)


double bench_t_start, bench_t_end;


// Функция для подсчета времени
static double rtclock() {
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0) {
        printf("Error return from gettimeofday: %d", stat);
    }
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}


// Инициализация времени старта
void bench_timer_start() {
    bench_t_start = rtclock();
}


// Инициализация времени финиша
void bench_timer_stop() {
    bench_t_end = rtclock();
}


// Вывод времени выполнения программы
void bench_timer_print() {
    printf("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}


// Инициализация массивов A и B
static void init_array (int n, double A[n][n][n], double B[n][n][n]) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
            	A[i][j][k] = B[i][j][k] = (double) (i + j + (n-k))* 10 / (n);
            }
        }
    }
}


// Печать массива
static void print_array(int n, double A[n][n][n]) {
    fprintf(stdout, "==BEGIN DUMP_ARRAYS==\n");
    fprintf(stdout, "begin dump: %s\n", "A");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                // Фрагмент, ставящий перенос строки после каждой строки, если N=20
//                 if ((i * n * n + j * n + k) % 20 == 0) {
//                     fprintf(stderr, "\n");
                 }
                // Заменим его, просто сделав перенос строки после завершения внутреннего цикла.
                fprintf(stdout, "%0.2lf ", A[i][j][k]);
            }
            fprintf(stdout, "\n");
        }
    }
    fprintf(stdout, "\nend   dump: %s\n", "A");
    fprintf(stdout, "==END   DUMP_ARRAYS==\n");
}


// Последовательная - изначальная
static void kernel_heat_3d(int tsteps, int n, double A[n][n][n], double B[n][n][n]) {
    for (int t = 1; t <= tsteps; t++) { // была ошибка TSTEPS

        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                for (int k = 1; k < n - 1; k++) {
                    B[i][j][k] = 0.125 * (A[i + 1][j][k] - 2.0 * A[i][j][k] + A[i - 1][j][k])
                                    + 0.125 * (A[i][j + 1][k] - 2.0 * A[i][j][k] + A[i][j - 1][k])
                                    + 0.125 * (A[i][j][k + 1] - 2.0 * A[i][j][k] + A[i][j][k - 1])
                                    + A[i][j][k];
                }
            }
        }

        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                for (int k = 1; k < n - 1; k++) {
                    A[i][j][k] = 0.125 * (B[i + 1][j][k] - 2.0 * B[i][j][k] + B[i - 1][j][k])
                                    + 0.125 * (B[i][j + 1][k] - 2.0 * B[i][j][k] + B[i][j - 1][k])
                                    + 0.125 * (B[i][j][k + 1] - 2.0 * B[i][j][k] + B[i][j][k - 1])
                                    + B[i][j][k];
                }
            }
        }
    }
}


// Параллельная - базовая
static void kernel_heat_3d_parallel_base(int tsteps, int n, double A[n][n][n], double B[n][n][n]) {
    int t, i, j, k;

    for (t = 1; t <= tsteps; t++) {

    	#pragma omp parallel for private(i, j, k)
        for (i = 1; i < n - 1; i++) {
            for (j = 1; j < n - 1; j++) {
                for (k = 1; k < n - 1; k++) {
                    double residue;
                    B[i][j][k] = A[i][j][k] * 0.25;
                    residue = A[i + 1][j][k] + A[i - 1][j][k] + A[i][j + 1][k] + A[i][j - 1][k]
                              + A[i][j][k + 1] + A[i][j][k - 1];
                    residue *= 0.125;
                    B[i][j][k] += residue;
                }
            }
        }

        #pragma omp parallel for private(i, j, k) 
        for (i = 1; i < n - 1; i++) {
            for (j = 1; j < n - 1; j++) {
                for (k = 1; k < n - 1; k++) {
                    double residue;
                    A[i][j][k] = B[i][j][k] * 0.25;
                    residue = B[i + 1][j][k] + B[i - 1][j][k] + B[i][j + 1][k] + B[i][j - 1][k]
                              + B[i][j][k + 1] + B[i][j][k - 1];
                    residue *= 0.125;
                    A[i][j][k] += residue;
                }
            }
        }
    }
}


// Параллельная - для небольших датасетов
static void kernel_heat_3d_parallel_mini(int tsteps, int n, double A[n][n][n], double B[n][n][n]) {
    int t, i, j, k;

    for (t = 1; t <= tsteps; t++) {

        #pragma omp parallel for private(i, j, k)
        for (i = 1; i < n - 1; i++) {
            for (j = 1; j < n - 1; j++) {
                for (k = 1; k + 3 < n - 1; k+=4) {
                    double residue;
                    B[i][j][k] = A[i][j][k] * 0.25;
                    residue = A[i + 1][j][k] + A[i - 1][j][k] + A[i][j + 1][k] + A[i][j - 1][k]
                                + A[i][j][k + 1] + A[i][j][k - 1];
                    residue *= 0.125;
                    B[i][j][k] += residue;

                    B[i][j][k + 1] = A[i][j][k + 1] * 0.25;
                    residue = A[i + 1][j][k + 1] + A[i - 1][j][k + 1] + A[i][j + 1][k + 1] + A[i][j - 1][k + 1]
                                + A[i][j][k + 2] + A[i][j][k];
                    residue *= 0.125;
                    B[i][j][k + 1] += residue;

                    B[i][j][k + 2] = A[i][j][k + 2] * 0.25;
                    residue = A[i + 1][j][k + 2] + A[i - 1][j][k + 2] + A[i][j + 1][k + 2] + A[i][j - 1][k + 2]
                                + A[i][j][k + 3] + A[i][j][k + 1];
                    residue *= 0.125;
                    B[i][j][k + 2] += residue;

                    B[i][j][k + 3] = A[i][j][k + 3] * 0.25;
                    residue = A[i + 1][j][k + 3] + A[i - 1][j][k + 3] + A[i][j + 1][k + 3] + A[i][j - 1][k + 3]
                                + A[i][j][k + 4] + A[i][j][k + 2];
                    residue *= 0.125;
                    B[i][j][k + 3] += residue;
                }
            }
        }

        #pragma omp parallel for private(i, j, k)
        for (i = 1; i < n - 1; i++) {
            for (j = 1; j < n - 1; j++) {
                for (k = 1; k + 3 < n - 1; k+=4) {
                    double residue;
                    A[i][j][k] = B[i][j][k] * 0.25;
                    residue = B[i + 1][j][k] + B[i - 1][j][k] + B[i][j + 1][k] + B[i][j - 1][k]
                                + B[i][j][k + 1] + B[i][j][k - 1];
                    residue *= 0.125;
                    A[i][j][k] += residue;

                    A[i][j][k + 1] = B[i][j][k + 1] * 0.25;
                    residue = B[i + 1][j][k + 1] + B[i - 1][j][k + 1] + B[i][j + 1][k + 1] + B[i][j - 1][k + 1]
                                + B[i][j][k + 2] + B[i][j][k];
                    residue *= 0.125;
                    A[i][j][k + 1] += residue;

                    A[i][j][k + 2] = B[i][j][k + 2] * 0.25;
                    residue = B[i + 1][j][k + 2] + B[i - 1][j][k + 2] + B[i][j + 1][k + 2] + B[i][j - 1][k + 2]
                                + B[i][j][k + 3] + B[i][j][k + 1];
                    residue *= 0.125;
                    A[i][j][k + 2] += residue;

                    A[i][j][k + 3] = B[i][j][k + 3] * 0.25;
                    residue = B[i + 1][j][k + 3] + B[i - 1][j][k + 3] + B[i][j + 1][k + 3] + B[i][j - 1][k + 3]
                                + B[i][j][k + 4] + B[i][j][k + 2];
                    residue *= 0.125;
                    A[i][j][k + 3] += residue;
                }
            }
        }
    }
}


// Параллельная для больших датасетов при сравнительно небольшом числе нитей
static void kernel_heat_3d_parallel_normal(
        int tsteps, int n, double A[n][n][n], double B[n][n][n], int block_size) {
    int t, i, j, k, kk, jj;

    for (t = 1; t <= tsteps; t++) {

        #pragma omp parallel for private(i, j, k, kk, jj)
        for (jj = 1; jj < n - 1; jj += block_size) {
        for (i = 1; i < n - 1; i++) {
                for (kk = 1; kk < n - 1; kk += block_size) {
                    for (j = jj; j < MIN(n - 1, jj + block_size); j++) {
                        for (k = kk; k < MIN(n - 1, kk + block_size); k++) {
                            double residue;
                            B[i][j][k] = A[i][j][k] * 0.25;
                            residue = A[i + 1][j][k] + A[i - 1][j][k]
                                    + A[i][j + 1][k] + A[i][j - 1][k]
                                    + A[i][j][k + 1] + A[i][j][k - 1];
                            residue *= 0.125;
                            B[i][j][k] += residue;
                        }
                    }
                }
            }
        }

        #pragma omp parallel for private(i, j, k, kk, jj)
        for (jj = 1; jj < n - 1; jj += block_size) {
        for (i = 1; i < n - 1; i++) {
                for (kk = 1; kk < n - 1; kk += block_size) {
                    for (j = jj; j < MIN(n - 1, jj + block_size); j++) {
                        for (k = kk; k < MIN(n - 1, kk + block_size); k++) {
                            double residue;
                            A[i][j][k] = B[i][j][k] * 0.25;
                            residue = B[i + 1][j][k] + B[i - 1][j][k]
                                    + B[i][j + 1][k] + B[i][j - 1][k]
                                    + B[i][j][k + 1] + B[i][j][k - 1];
                            residue *= 0.125;
                            A[i][j][k] += residue;
                        }
                    }
                }
            }
        }
    }
}



// Параллельная для больших датасетов при сравнительно большом числе нитей
static void kernel_heat_3d_parallel_big(
        int tsteps, int n, double A[n][n][n], double B[n][n][n], int block_size) {
    int t, i, j, k, kk, jj;

    for (t = 1; t <= tsteps; t++) {

        #pragma omp parallel private(i, j, k, kk, jj)
        for (i = 1; i < n - 1; i++) {
            for (jj = 1; jj < n - 1; jj += block_size) {
                for (kk = 1; kk < n - 1; kk += block_size) {
                    for (j = jj; j < MIN(n - 1, jj + block_size); j++) {
                        for (k = kk; k < MIN(n - 1, kk + block_size); k++) {
                            double residue;
                            B[i][j][k] = A[i][j][k] * 0.25;
                            residue = A[i + 1][j][k] + A[i - 1][j][k]
                                    + A[i][j + 1][k] + A[i][j - 1][k]
                                    + A[i][j][k + 1] + A[i][j][k - 1];
                            residue *= 0.125;
                            B[i][j][k] += residue;
                        }
                    }
                }
            }
        }

        #pragma omp parallel private(i, j, k, kk, jj)
        for (i = 1; i < n - 1; i++) {
            for (jj = 1; jj < n - 1; jj += block_size) {
                for (kk = 1; kk < n - 1; kk += block_size) {
                    for (j = jj; j < MIN(n - 1, jj + block_size); j++) {
                        for (k = kk; k < MIN(n - 1, kk + block_size); k++) {
                            double residue;
                            A[i][j][k] = B[i][j][k] * 0.25;
                            residue = B[i + 1][j][k] + B[i - 1][j][k]
                                    + B[i][j + 1][k] + B[i][j - 1][k]
                                    + B[i][j][k + 1] + B[i][j][k - 1];
                            residue *= 0.125;
                            A[i][j][k] += residue;
                        }
                    }
                }
            }
        }
    }
}


static void parallel_program(int tsteps, int n, double A[n][n][n], double B[n][n][n],
                             int id_dataset, int block_size, int count_threads) {
    if (id_dataset == 1 || id_dataset == 2 || id_dataset == 3) {
        // небольшой датасет
        kernel_heat_3d_parallel_mini(tsteps, n, A, B);
    } else if ((id_dataset == 4 || id_dataset == 5) && count_threads <= n / block_size) {
        // большой датасет и число нитей позволяют распараллелить по блокам
        kernel_heat_3d_parallel_normal(tsteps, n, A, B, block_size);
    } else {
        // большой датасет, а число нитей позволяет распараллелить внешний цикл
        kernel_heat_3d_parallel_big(tsteps, n, A, B, block_size);
    }
}


int main(int argc, char** argv) {
    
    int count_threads = omp_get_max_threads();

	printf("\n*** START ***\n");
	printf("OMP_NUM_THREADS = %d", count_threads);
	printf("\n");

	int id_dataset = atoi(argv[1]); // идентификатор датасета
	int n; // размер оси 3х-мерной матрицы
	int tsteps; // число шагов в kernel_heat_3d
    int block_size = 0; // размер блока

	if (id_dataset == 1) {
		printf("MINI DATASET\n");
		n = 10;
		tsteps = 20;
	} else if (id_dataset == 2) {
		printf("SMALL DATASET\n");
		n = 20;
		tsteps = 40;
	} else if (id_dataset == 3) {
		printf("MEDIUM DATASET\n");
		n = 40;
		tsteps = 100;
	} else if (id_dataset == 4) {
		printf("LARGE DATASET\n");
		n = 120;
		tsteps = 500;
        block_size = 5;
	} else if (id_dataset == 5) {
		printf("EXTRALARGE DATASET\n");
		n = 200;
		tsteps = 1000;
        block_size = 10;
	}


    double (*A)[n][n][n];
    double (*B)[n][n][n];


    //********** Последовательная программа **********//
    A = (double(*)[n][n][n])malloc ((n) * (n) * (n) * sizeof(double));
    B = (double(*)[n][n][n])malloc ((n) * (n) * (n) * sizeof(double));
    init_array(n, *A, *B); // Иницилизация массивов A и B
    printf("Sequential program: ");
    double time_start = omp_get_wtime(); // Иницилизация времени старта
    kernel_heat_3d(tsteps, n, *A, *B); // Основная функция
    double time_finish = omp_get_wtime(); // Иницилизация времени финиша
    double time_execution = time_finish - time_start;
    printf("time in seconds = %0.6lf\n", time_execution); // Вывод времени выполнения программы
    free((void*)A);
    free((void*)B);


    //********** Параллельная программа **********//
    A = (double(*)[n][n][n])malloc ((n) * (n) * (n) * sizeof(double));
    B = (double(*)[n][n][n])malloc ((n) * (n) * (n) * sizeof(double));
    init_array(n, *A, *B); // Иницилизация массивов A и B
    printf("Parallel program: ");
    double time_start = omp_get_wtime(); // Иницилизация времени старта
    parallel_program(tsteps, n, *A, *B, id_dataset, block_size, count_threads); // Основная функция
    double time_finish = omp_get_wtime(); // Иницилизация времени финиша
    double time_execution = time_finish - time_start;
    printf("time in seconds = %0.6lf\n", time_execution); // Вывод времени выполнения программы
    free((void*)A);
    free((void*)B);


    //********** ДЛЯ ТЕСТИРОВАНИЯ **********//
//    double parallel[5][6], sequential[6];
//
//    for (int t = 1; t <= 4; t++) {
//        for (int i = 1; i <= 5; i++) {
//            parallel[t][i] = 0;
//        }
//    }
//    for (int i = 1; i <= 5; i++) {
//            sequential[i] = 0;
//    }
//
//    for (int k = 1; k <= 10; k++) {
//        for (int id_dataset = 1; id_dataset <= 5; id_dataset++) {
//            if (id_dataset == 1) {
//                printf("MINI DATASET\n");
//                n = 10;
//                tsteps = 20;
//            } else if (id_dataset == 2) {
//                printf("SMALL DATASET\n");
//                n = 20;
//                tsteps = 40;
//            } else if (id_dataset == 3) {
//                printf("MEDIUM DATASET\n");
//                n = 40;
//                tsteps = 100;
//            } else if (id_dataset == 4) {
//                printf("LARGE DATASET\n");
//                n = 120;
//                tsteps = 500;
//                block_size = 5;
//            } else if (id_dataset == 5) {
//                printf("EXTRALARGE DATASET\n");
//                n = 200;
//                tsteps = 1000;
//                block_size = 10;
//            }
//            A = (double(*)[n][n][n])malloc ((n) * (n) * (n) * sizeof(double));
//            B = (double(*)[n][n][n])malloc ((n) * (n) * (n) * sizeof(double));
//            init_array(n, *A, *B); // Иницилизация массивов A и B
//            printf("Sequential program: ");
//            double time_start = omp_get_wtime(); // Иницилизация времени старта
//            kernel_heat_3d(tsteps, n, *A, *B); // Основная функция
//            double time_finish = omp_get_wtime(); // Иницилизация времени финиша
//            double time_execution = time_finish - time_start;
//            sequential[id_dataset] += time_execution;
//            // printf("time in seconds = %0.6lf\n", time_execution); // Вывод времени выполнения программы
//            free((void*)A);
//            free((void*)B);
//        }
//    }
//
//    for (int k = 1; k <= 10; k++) {
//        for (int t = 1; t <= count_threads; t++) {
//            omp_set_num_threads(t);
//            printf("Текущее число нитей = %d\n", t);
//            for (int id_dataset = 1; id_dataset <= 5; id_dataset++) {
//                if (id_dataset == 1) {
//                    printf("MINI DATASET\n");
//                    n = 10;
//                    tsteps = 20;
//                } else if (id_dataset == 2) {
//                    printf("SMALL DATASET\n");
//                    n = 20;
//                    tsteps = 40;
//                } else if (id_dataset == 3) {
//                    printf("MEDIUM DATASET\n");
//                    n = 40;
//                    tsteps = 100;
//                } else if (id_dataset == 4) {
//                    printf("LARGE DATASET\n");
//                    n = 120;
//                    tsteps = 500;
//                    block_size = 5;
//                } else if (id_dataset == 5) {
//                    printf("EXTRALARGE DATASET\n");
//                    n = 200;
//                    tsteps = 1000;
//                    block_size = 10;
//                }
//                A = (double(*)[n][n][n])malloc ((n) * (n) * (n) * sizeof(double));
//                B = (double(*)[n][n][n])malloc ((n) * (n) * (n) * sizeof(double));
//                init_array(n, *A, *B); // Иницилизация массивов A и B
//                printf("Parallel program: ");
//                double time_start = omp_get_wtime(); // Иницилизация времени старта
//                parallel_program(tsteps, n, *A, *B, id_dataset, block_size, count_threads); // Основная функция
//                double time_finish = omp_get_wtime(); // Иницилизация времени финиша
//                double time_execution = time_finish - time_start;
//                parallel[t][id_dataset] += time_execution;
//                // printf("time in seconds = %0.6lf\n", time_execution); // Вывод времени выполнения программы
//                free((void*)A);
//                free((void*)B);
//            }
//        }
//    }
//    for (int t = 1; t <= 4; t++) {
//        for (int i = 1; i <= 5; i++) {
//            printf("dataset %d with %d treads: seg: %0.6lf ; parallel: %0.6lf ; \n", i, t, sequential[i] / 10, parallel[t][i] / 10);
//        }
//    }

    printf("*** FINISH ***\n");
    return 0;
}
