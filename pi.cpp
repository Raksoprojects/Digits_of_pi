#include <iostream>
#include <math.h>
#include <ctime>
#include <chrono>
#include <omp.h>
#include <iomanip>

#define N 134217728
#define PI 3.141592653589793

double *dcreate_mat(){
    double *d = new double[N];
    for(int i=0;i<N;++i ){
        d[i] = 4 * (pow(-1,i)/(2*(i+1) - 1));
    }
    return d;
}

float *fcreate_mat(){
    float *d = new float[N];
    for(int i=0;i<N;++i ){
        d[i] = 4 * (pow(-1,i)/(2*(i+1) - 1));
    }
    return d;
}

//==========================CPU part========================
double dleibniz_cpu(double * tab){

    double suma = 0;
    for(int i=0;i<N;++i){
        suma += tab[i];
    }
    return suma;
}
float fleibniz_cpu(float * tab){

    float suma = 0;
    for(int i=0;i<N;++i){
        suma += tab[i];
    }
    return suma;
}

double dleibniz_cpu_mp(double * tab){

    int i=0;
    int j=0;
    
    for(i=N/2;i>0;i=i/2){
        #pragma omp parallel for private(j) shared(tab) ordered
        for(j=0;j<i;j++){
            tab[j] = tab[j] + tab[i+j];
        }
    }
    return tab[0];
}
float fleibniz_cpu_mp(float * tab){

    int i=0;
    int j=0;
    
    for(i=N/2;i>=1;i=i/2){
        #pragma omp parallel for private(j) shared(tab) ordered
        for(j=0;j<i;j++){
            tab[j] = tab[j] + tab[i+j];
        }
    }
    return tab[0];
}

int main(){
    //g++ -g -o macierzCPU macierzCPU.cpp -fopenmp  do skompilowania we standardzie openMP
    std::cout<<"Pi: "<<std::setprecision(15)<<PI<<'\n';
    double dsum_cpu;
    double *dtab = dcreate_mat();
    float *ftab = fcreate_mat();
    auto t_start = std::chrono::high_resolution_clock::now();
    dsum_cpu = dleibniz_cpu(dtab);
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    std::cout<<"Pi na cpu(double): "<<std::setprecision(15)<<dsum_cpu<<'\n';
    std::cout<<"It took: "<< std::setprecision(20)<<elapsed_time_ms << " ms"<<"\n\n";

    float fsum_cpu ;
    t_start = std::chrono::high_resolution_clock::now();
    fsum_cpu = fleibniz_cpu(ftab);
    t_end = std::chrono::high_resolution_clock::now();
    elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    std::cout<<"Pi na cpu(float): "<<std::setprecision(15)<<fsum_cpu<<'\n';
    std::cout<<"It took: "<<std::setprecision(20)<< elapsed_time_ms << " ms"<<"\n\n";
    
    omp_set_num_threads(omp_get_num_procs()); //używanie maksymalnej liczby wątków
    double dsum_cpu_mp;
    float fsum_cpu_mp;
    t_start = std::chrono::high_resolution_clock::now();
    dsum_cpu_mp = dleibniz_cpu_mp(dtab);
    t_end = std::chrono::high_resolution_clock::now();
    elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    std::cout<<"Pi na cpu openmp(double): "<<std::setprecision(15)<<dsum_cpu_mp<<'\n';
    std::cout<<"It took: "<<std::setprecision(20)<< elapsed_time_ms << " ms"<<"\n\n";

    t_start = std::chrono::high_resolution_clock::now();
    fsum_cpu_mp = fleibniz_cpu_mp(ftab);
    t_end = std::chrono::high_resolution_clock::now();
    elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    std::cout<<"Pi na cpu openmp(float): "<<std::setprecision(15)<<dsum_cpu_mp<<'\n';
    std::cout<<"It took: "<<std::setprecision(20)<< elapsed_time_ms << " ms"<<"\n\n";

    delete [] dtab;
    delete [] ftab;
}