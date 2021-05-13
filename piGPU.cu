#include <iostream>
#include <math.h>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <omp.h>

#define N 134217728
#define THREADS 128
#define PI 3.141592653589793

//==========================================matrix creation=====================================================
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



//==========================CPU part========================================================
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

//==========================CPU OpenMP part========================================================
//Niestety nie wiem jak skompilować to na cudzie, a raczej na ts-tigerze, natomiast u mnie na komputerze działa
//przy pomocy komendy: g++ -g -o macierzCPU macierzCPU.cpp -fopenmp
//#pragma omp parallel for private(i) shared(macierzC)
// double dleibniz_cpu_mp(double * tab){

//     double suma = 0;
//     #pragma omp parallel for private(i) shared(tab)
//     for(int i=0;i<N;++i){
//         suma += tab[i];
//     }
//     return suma;
// }
// float fleibniz_cpu_mp(float * tab){

//     float suma = 0;
//     #pragma omp parallel for private(i) shared(tab)
//     for(int i=0;i<N;++i){
//         suma += tab[i];
//     }
//     return suma;
// }

//============================GPU part ==================================================

__global__ void dleibniz_gpu(double *tab, double *dsum){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //__syncthreads();
    for(int i = N/2;i>0;i=i/2){
        //__syncthreads();
        if(idx < i){ 
            //__syncthreads(); 
            tab[idx] = tab[idx] + tab[i+idx];   
        //__syncthreads(); 
        }
    }
    *dsum = tab[0];
}

__global__ void fleibniz_gpu(float *tab, float *fsum){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //__syncthreads();
    
    for(int i = N/2;i>=1;i=i/2){
        //__syncthreads();
        if(idx < i){
            //__syncthreads();     
            tab[idx] = tab[idx] + tab[i+idx];   
        //__syncthreads(); 
        }
    }
    *fsum = tab[0];
}

int main(){
    
    std::cout<<"Pi: "<<std::setprecision(15)<<PI<<'\n';
    double dsum_cpu;
    double *dtab = dcreate_mat();
    float *ftab = fcreate_mat();
    auto t_start = std::chrono::high_resolution_clock::now();
    dsum_cpu = dleibniz_cpu(dtab);
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    std::cout<<"Pi na cpu(double): "<<std::setprecision(15)<<dsum_cpu<<'\n';
    std::cout<<"Błąd: "<<100*(PI-dsum_cpu)/dsum_cpu<<"%"<<'\n';
    std::cout<<"It took: "<< elapsed_time_ms << " ms"<<"\n\n";

    float fsum_cpu ;
    t_start = std::chrono::high_resolution_clock::now();
    fsum_cpu = fleibniz_cpu(ftab);
    t_end = std::chrono::high_resolution_clock::now();
    elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    std::cout<<"Pi na cpu(float): "<<std::setprecision(15)<<fsum_cpu<<'\n';
    std::cout<<"Błąd: "<<100*(PI-fsum_cpu)/fsum_cpu<<"%"<<'\n';
    std::cout<<"It took: "<< elapsed_time_ms << " ms"<<"\n\n";

    //==================================CPU OpenMP====================================================
    //omp_set_num_threads(omp_get_num_procs());
    // double dsum_cpu_mp;
    // float fsum_cpu_mp;
    // t_start = std::chrono::high_resolution_clock::now();
    // dsum_cpu_mp = dleibniz_cpu(dtab);
    // t_end = std::chrono::high_resolution_clock::now();
    // elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    // std::cout<<"Pi na cpu openmp(double): "<<std::setprecision(15)<<dsum_cpu_mp<<'\n';
    // std::cout<<"It took: "<< elapsed_time_ms << " ms"<<"\n\n";

    // t_start = std::chrono::high_resolution_clock::now();
    // fsum_cpu_mp = dleibniz_cpu(ftab);
    // t_end = std::chrono::high_resolution_clock::now();
    // elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    // std::cout<<"Pi na cpu openmp(float): "<<std::setprecision(15)<<dsum_cpu_mp<<'\n';
    // std::cout<<"It took: "<< elapsed_time_ms << " ms"<<"\n\n";


    //=================================GPU==================================
    //double *dtab = dcreate_mat();
    
    double *d_dtab; //device tablica double
    size_t size_d = sizeof(double) * N;
    double dsum_gpu; 
    double *d_dsum_gpu;

    cudaMalloc(&d_dtab,size_d);
    cudaMalloc((void **)&d_dsum_gpu,sizeof(double));
    
    cudaMemcpy(d_dtab, dtab, size_d, cudaMemcpyHostToDevice);
    t_start = std::chrono::high_resolution_clock::now();
    dleibniz_gpu<<<ceil(N/float(THREADS)), THREADS>>>(d_dtab,d_dsum_gpu);
    t_end = std::chrono::high_resolution_clock::now();
    elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    cudaMemcpy(&dsum_gpu, d_dsum_gpu, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_dtab); cudaFree(d_dsum_gpu);
    cudaDeviceReset();

    std::cout<<"Pi na gpu(double): "<<std::setprecision(15)<<dsum_gpu<<'\n';
    std::cout<<"Błąd: "<<100*(PI-dsum_gpu)/dsum_gpu<<"%"<<'\n';
    std::cout<<"It took: "<< elapsed_time_ms << " ms"<<"\n\n";
    
    //==================float============== 
    float *d_ftab; //device tablica float
    size_t size_f = sizeof(float) * N;
    //float part
    float fsum_gpu;
    float *d_fsum_gpu;

    cudaMalloc(&d_ftab,size_f);
    cudaMalloc((void **)&d_fsum_gpu,sizeof(float));
    
    cudaMemcpy(d_ftab, ftab, size_f, cudaMemcpyHostToDevice);
    t_start = std::chrono::high_resolution_clock::now();
    fleibniz_gpu<<<ceil(N/float(THREADS)), THREADS>>>(d_ftab,d_fsum_gpu);
    t_end = std::chrono::high_resolution_clock::now();
    elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    cudaMemcpy(&fsum_gpu, d_fsum_gpu, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_ftab); cudaFree(d_fsum_gpu);
    cudaDeviceReset();

    std::cout<<"Pi na gpu(float): "<<std::setprecision(15)<<fsum_gpu<<'\n';
    std::cout<<"Błąd: "<<100*(PI-fsum_gpu)/fsum_gpu<<"%"<<'\n';
    std::cout<<"It took: "<< elapsed_time_ms << " ms"<<"\n\n";

    delete [] dtab;
    delete [] ftab;
    //delete [] d_dtab;
    //delete [] d_ftab;


    return 0;
}