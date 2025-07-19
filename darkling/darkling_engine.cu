#include <stdint.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_MASK_LEN 32
#define MAX_CHARSET_SIZE 128
#define MAX_HASHES 1024

// constant buffers for mask charsets and hashes
__constant__ char d_charsets[MAX_MASK_LEN][MAX_CHARSET_SIZE];
__constant__ int  d_charset_lens[MAX_MASK_LEN];
__constant__ uint8_t d_hashes[MAX_HASHES][20];   // supports up to SHA1
__constant__ int d_num_hashes;
__constant__ int d_hash_len;  // digest length (16 for MD5, 20 for SHA1)
__constant__ int d_pwd_len;

__device__ inline uint32_t rotl32(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

// simple MD5 implementation for short inputs (<55 bytes)
__device__ void md5(const char *msg, int len, uint8_t out[16]) {
    const uint32_t init[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
    uint32_t a = init[0], b = init[1], c = init[2], d = init[3];

    uint8_t buffer[64];
    for (int i=0;i<64;i++) buffer[i] = 0;
    for (int i=0;i<len;i++) buffer[i] = (uint8_t)msg[i];
    buffer[len] = 0x80;
    uint64_t bits = (uint64_t)len * 8;
    for (int i=0;i<8;i++) buffer[56+i] = (bits >> (8*i)) & 0xff;

    const uint32_t k[] = {
        0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
        0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
        0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
        0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
        0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
        0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
        0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
        0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
    };
    const uint32_t r[] = {
        7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
        5,9,14,20,5,9,14,20,5,9,14,20,5,9,14,20,
        4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
        6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21
    };

    uint32_t w[16];
    for(int i=0;i<16;i++) {
        w[i] = (uint32_t)buffer[i*4] | ((uint32_t)buffer[i*4+1]<<8) |
               ((uint32_t)buffer[i*4+2]<<16) | ((uint32_t)buffer[i*4+3]<<24);
    }
    uint32_t aa=a, bb=b, cc=c, dd=d;
    for(int i=0;i<64;i++) {
        uint32_t f,g;
        if(i<16)      { f=(b & c) | (~b & d); g=i; }
        else if(i<32) { f=(d & b) | (~d & c); g=(5*i+1)&15; }
        else if(i<48) { f=b ^ c ^ d; g=(3*i+5)&15; }
        else          { f=c ^ (b | ~d); g=(7*i)&15; }
        uint32_t temp=d; d=c; c=b;
        uint32_t t=a + f + k[i] + w[g];
        b += rotl32(t, r[i]);
        a=temp;
    }
    a+=aa; b+=bb; c+=cc; d+=dd;
    uint32_t out32[4]={a,b,c,d};
    for(int i=0;i<4;i++){
        out[i*4]=(out32[i])&0xff;
        out[i*4+1]=(out32[i]>>8)&0xff;
        out[i*4+2]=(out32[i]>>16)&0xff;
        out[i*4+3]=(out32[i]>>24)&0xff;
    }
}

// simple SHA1 implementation for short inputs (<56 bytes)
__device__ void sha1(const char *msg, int len, uint8_t out[20]) {
    uint32_t h0=0x67452301, h1=0xefcdab89, h2=0x98badcfe, h3=0x10325476, h4=0xc3d2e1f0;
    uint8_t buffer[64];
    for(int i=0;i<64;i++) buffer[i]=0;
    for(int i=0;i<len;i++) buffer[i]=(uint8_t)msg[i];
    buffer[len]=0x80;
    uint64_t bits=(uint64_t)len*8;
    for(int i=0;i<8;i++) buffer[56+i]=(bits>>(56-8*i))&0xff;
    uint32_t w[80];
    for(int i=0;i<16;i++) {
        w[i]=((uint32_t)buffer[4*i]<<24)|((uint32_t)buffer[4*i+1]<<16)|((uint32_t)buffer[4*i+2]<<8)|buffer[4*i+3];
    }
    for(int i=16;i<80;i++) w[i]=rotl32(w[i-3]^w[i-8]^w[i-14]^w[i-16],1);
    uint32_t a=h0,b=h1,c=h2,d=h3,e=h4;
    for(int i=0;i<80;i++) {
        uint32_t f,k;
        if(i<20){f=(b&c)|((~b)&d);k=0x5a827999;}
        else if(i<40){f=b^c^d;k=0x6ed9eba1;}
        else if(i<60){f=(b&c)|(b&d)|(c&d);k=0x8f1bbcdc;}
        else{f=b^c^d;k=0xca62c1d6;}
        uint32_t temp=rotl32(a,5)+f+e+k+w[i];
        e=d; d=c; c=rotl32(b,30); b=a; a=temp;
    }
    h0+=a; h1+=b; h2+=c; h3+=d; h4+=e;
    uint32_t hv[5]={h0,h1,h2,h3,h4};
    for(int i=0;i<5;i++){
        out[i*4]=(hv[i]>>24)&0xff;
        out[i*4+1]=(hv[i]>>16)&0xff;
        out[i*4+2]=(hv[i]>>8)&0xff;
        out[i*4+3]=hv[i]&0xff;
    }
}

__device__ void compute_hash(const char *pwd, int len, uint8_t *out) {
    if (d_hash_len == 16) md5(pwd, len, out);
    else sha1(pwd, len, out);
}

__device__ bool check_hash(const uint8_t *digest) {
    for (int h=0; h<d_num_hashes; ++h) {
        bool match = true;
        for(int i=0;i<d_hash_len;i++) {
            if (digest[i] != d_hashes[h][i]) { match = false; break; }
        }
        if (match) return true;
    }
    return false;
}

__global__ void crack_kernel(uint64_t total, char *results, int max_results, int *found_count) {
    uint64_t idx = threadIdx.x + blockIdx.x * (uint64_t)blockDim.x;
    uint64_t step = gridDim.x * (uint64_t)blockDim.x;
    char pwd[MAX_MASK_LEN];
    uint8_t digest[20];

    while (idx < total) {
        uint64_t val = idx;
        for (int pos = d_pwd_len - 1; pos >= 0; --pos) {
            int len = d_charset_lens[pos];
            int c = val % len;
            pwd[pos] = d_charsets[pos][c];
            val /= len;
        }
        compute_hash(pwd, d_pwd_len, digest);
        if (check_hash(digest)) {
            int slot = atomicAdd(found_count, 1);
            if (slot < max_results) {
                for(int i=0;i<d_pwd_len;i++) results[slot*MAX_MASK_LEN + i] = pwd[i];
                results[slot*MAX_MASK_LEN + d_pwd_len] = '\0';
            }
        }
        idx += step;
    }
}

extern "C" void launch_darkling(const char **charsets, const int *lens, int pwd_len,
                                 const uint8_t *hashes, int num_hashes, int hash_len,
                                 uint64_t total, char *d_results, int max_results, int *d_count,
                                 dim3 grid, dim3 block)
{
    cudaMemcpyToSymbol(d_pwd_len, &pwd_len, sizeof(int));
    cudaMemcpyToSymbol(d_hash_len, &hash_len, sizeof(int));
    cudaMemcpyToSymbol(d_num_hashes, &num_hashes, sizeof(int));
    for(int i=0;i<pwd_len;i++) {
        cudaMemcpyToSymbol(d_charsets[i], charsets[i], lens[i], 0);
        cudaMemcpyToSymbol(d_charset_lens[i], &lens[i], sizeof(int));
    }
    cudaMemcpyToSymbol(d_hashes, hashes, num_hashes * hash_len);
    crack_kernel<<<grid, block>>>(total, d_results, max_results, d_count);
}
