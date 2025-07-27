#include "intel_cracker.h"
#include <CL/cl.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <algorithm>

#define MAX_CUSTOM_SETS 16
#define MAX_CHARSET_CHARS 256
#define MAX_UTF8_BYTES 4
#define MAX_PWD_BYTES (MAX_MASK_LEN * MAX_UTF8_BYTES)

namespace darkling {

IntelCracker::IntelCracker() {}
IntelCracker::~IntelCracker() {}

bool IntelCracker::initialize() {
    return true;
}

bool IntelCracker::load_job(const MaskJob &job) {
    job_ = job;
    return true;
}

bool IntelCracker::run_batch() {
    start_time_ = std::chrono::high_resolution_clock::now();
    cl_uint num=0;
    if(clGetPlatformIDs(0,nullptr,&num)!=CL_SUCCESS||num==0) return false;
    std::vector<cl_platform_id> plats(num);
    clGetPlatformIDs(num, plats.data(), nullptr);
    cl_device_id dev;
    clGetDeviceIDs(plats[0], CL_DEVICE_TYPE_DEFAULT,1,&dev,nullptr);
    cl_int err=0;
    cl_context ctx=clCreateContext(nullptr,1,&dev,nullptr,nullptr,&err);
    cl_command_queue q=clCreateCommandQueue(ctx,dev,0,&err);

    std::ifstream file("darkling/intel_backend/opencl_kernel.cl");
    std::string src((std::istreambuf_iterator<char>(file)),{});
    const char* sc=src.c_str(); size_t sl=src.size();
    cl_program prog=clCreateProgramWithSource(ctx,1,&sc,&sl,&err);
    clBuildProgram(prog,0,nullptr,nullptr,nullptr,nullptr);
    cl_kernel kr=clCreateKernel(prog,"crack_kernel",&err);

    static uint8_t cs_bytes[MAX_CUSTOM_SETS][MAX_CHARSET_CHARS][MAX_UTF8_BYTES];
    static uint8_t cs_lens[MAX_CUSTOM_SETS][MAX_CHARSET_CHARS];
    static int cs_sizes[MAX_CUSTOM_SETS];
    for(int s=0;s<MAX_CUSTOM_SETS;s++){
        int len=job_.charset_lengths[s];
        cs_sizes[s]=len;
        for(int i=0;i<len;i++){ cs_bytes[s][i][0]=job_.charsets[s][i]; cs_lens[s][i]=1; }
    }

    cl_mem pos_buf=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,job_.mask_length,job_.mask_template,&err);
    cl_mem bytes_buf=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(cs_bytes),cs_bytes,&err);
    cl_mem len_buf=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(cs_lens),cs_lens,&err);
    cl_mem size_buf=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(cs_sizes),cs_sizes,&err);
    size_t hash_sz=job_.num_hashes*job_.hash_length;
    cl_mem hash_buf=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,hash_sz,job_.hashes,&err);
    cl_mem res_buf=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,MAX_RESULT_BUFFER*MAX_PWD_BYTES,nullptr,&err);
    int zero = 0;
    cl_mem cnt_buf=clCreateBuffer(ctx,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(int),&zero,&err);

    int idx=0;
    clSetKernelArg(kr,idx++,sizeof(cl_ulong),&job_.start_index);
    cl_ulong total=job_.end_index-job_.start_index; clSetKernelArg(kr,idx++,sizeof(cl_ulong),&total);
    clSetKernelArg(kr,idx++,sizeof(cl_mem),&pos_buf);
    clSetKernelArg(kr,idx++,sizeof(cl_mem),&bytes_buf);
    clSetKernelArg(kr,idx++,sizeof(cl_mem),&len_buf);
    clSetKernelArg(kr,idx++,sizeof(cl_mem),&size_buf);
    clSetKernelArg(kr,idx++,sizeof(cl_mem),&hash_buf);
    clSetKernelArg(kr,idx++,sizeof(int),&job_.num_hashes);
    clSetKernelArg(kr,idx++,sizeof(int),&job_.hash_length);
    clSetKernelArg(kr,idx++,sizeof(cl_uchar),&job_.hash_type);
    clSetKernelArg(kr,idx++,sizeof(int),&job_.mask_length);
    clSetKernelArg(kr,idx++,sizeof(cl_mem),&res_buf);
    int max_results_const = MAX_RESULT_BUFFER;
    clSetKernelArg(kr,idx++,sizeof(int),&max_results_const);
    clSetKernelArg(kr,idx++,sizeof(cl_mem),&cnt_buf);

    size_t global=64;
    clEnqueueNDRangeKernel(q,kr,1,nullptr,&global,nullptr,0,nullptr,nullptr);
    clFinish(q);

    int h_count=0;
    clEnqueueReadBuffer(q,cnt_buf,CL_TRUE,0,sizeof(int),&h_count,0,nullptr,nullptr);
    h_count = std::min(h_count, MAX_RESULT_BUFFER);
    std::vector<char> buffer(static_cast<size_t>(h_count)*MAX_PWD_BYTES);
    if(h_count>0)
        clEnqueueReadBuffer(q,res_buf,CL_TRUE,0,buffer.size(),buffer.data(),0,nullptr,nullptr);

    results_.clear();
    for(int i=0;i<h_count;i++) {
        const char* pwd=buffer.data()+static_cast<size_t>(i)*MAX_PWD_BYTES;
        CrackResult r{};
        r.candidate_index = 0;
        r.length = static_cast<uint8_t>(std::strlen(pwd));
        std::memcpy(r.password,pwd,r.length);
        std::memset(r.hash,0,sizeof(r.hash));
        results_.push_back(r);
    }

    end_time_ = std::chrono::high_resolution_clock::now();

    clReleaseMemObject(pos_buf);
    clReleaseMemObject(bytes_buf);
    clReleaseMemObject(len_buf);
    clReleaseMemObject(size_buf);
    clReleaseMemObject(hash_buf);
    clReleaseMemObject(res_buf);
    clReleaseMemObject(cnt_buf);
    clReleaseKernel(kr);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    return true;
}

std::vector<CrackResult> IntelCracker::read_results() {
    auto out = results_;
    results_.clear();
    return out;
}

GpuStatus IntelCracker::get_status() {
    GpuStatus s{};
    s.hashes_processed = job_.end_index - job_.start_index;
    s.batch_duration_ms = std::chrono::duration<float,std::milli>(end_time_ - start_time_).count();
    s.gpu_temp_c = 0.0f;
    s.overheat_flag = false;
    return s;
}

} // namespace darkling
