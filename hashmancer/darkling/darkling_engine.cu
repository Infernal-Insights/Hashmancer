#include <stdint.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "darkling_engine.h"

// constant buffers for mask charsets and hashes
__constant__ uint8_t d_charset_bytes[MAX_CUSTOM_SETS][MAX_CHARSET_CHARS][MAX_UTF8_BYTES];
__constant__ uint8_t d_charset_charlen[MAX_CUSTOM_SETS][MAX_CHARSET_CHARS];
__constant__ int  d_charset_lens[MAX_CUSTOM_SETS];
__constant__ uint8_t d_pos_charset[MAX_MASK_LEN];
__constant__ uint8_t d_hashes[MAX_HASHES][20];   // supports up to SHA1
__constant__ int d_num_hashes;
__constant__ int d_hash_len;  // digest length (16 for MD5, 20 for SHA1)
__constant__ int d_pwd_len;
__constant__ uint8_t d_hash_type;  // 1=MD5,2=SHA1,3=NTLM

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

// simple MD4 used for NTLM (expects UTF-16LE input)
__device__ void md4(const uint8_t *msg, int len, uint8_t out[16]) {
    uint32_t a=0x67452301, b=0xefcdab89, c=0x98badcfe, d=0x10325476;
    uint8_t buffer[64];
    for(int i=0;i<64;i++) buffer[i]=0;
    for(int i=0;i<len;i++) buffer[i]=msg[i];
    buffer[len]=0x80;
    uint64_t bits=(uint64_t)len*8;
    for(int i=0;i<8;i++) buffer[56+i]=(bits>>(8*i))&0xff;
    uint32_t w[16];
    for(int i=0;i<16;i++)
        w[i]=((uint32_t)buffer[i*4])|((uint32_t)buffer[i*4+1]<<8)|((uint32_t)buffer[i*4+2]<<16)|((uint32_t)buffer[i*4+3]<<24);
#define F(x,y,z) ((x & y) | (~x & z))
#define G(x,y,z) ((x & y) | (x & z) | (y & z))
#define H(x,y,z) (x ^ y ^ z)
#define ROUND(a,b,c,d,k,s,func,add) a = rotl32(a + func(b,c,d) + w[k] + add, s)
    ROUND(a,b,c,d,0,3,F,0);  ROUND(d,a,b,c,1,7,F,0);  ROUND(c,d,a,b,2,11,F,0); ROUND(b,c,d,a,3,19,F,0);
    ROUND(a,b,c,d,4,3,F,0);  ROUND(d,a,b,c,5,7,F,0);  ROUND(c,d,a,b,6,11,F,0); ROUND(b,c,d,a,7,19,F,0);
    ROUND(a,b,c,d,8,3,F,0);  ROUND(d,a,b,c,9,7,F,0);  ROUND(c,d,a,b,10,11,F,0); ROUND(b,c,d,a,11,19,F,0);
    ROUND(a,b,c,d,12,3,F,0); ROUND(d,a,b,c,13,7,F,0); ROUND(c,d,a,b,14,11,F,0); ROUND(b,c,d,a,15,19,F,0);
    ROUND(a,b,c,d,0,3,G,0x5a827999);  ROUND(d,a,b,c,4,5,G,0x5a827999);  ROUND(c,d,a,b,8,9,G,0x5a827999);  ROUND(b,c,d,a,12,13,G,0x5a827999);
    ROUND(a,b,c,d,1,3,G,0x5a827999);  ROUND(d,a,b,c,5,5,G,0x5a827999);  ROUND(c,d,a,b,9,9,G,0x5a827999);  ROUND(b,c,d,a,13,13,G,0x5a827999);
    ROUND(a,b,c,d,2,3,G,0x5a827999);  ROUND(d,a,b,c,6,5,G,0x5a827999);  ROUND(c,d,a,b,10,9,G,0x5a827999); ROUND(b,c,d,a,14,13,G,0x5a827999);
    ROUND(a,b,c,d,3,3,G,0x5a827999);  ROUND(d,a,b,c,7,5,G,0x5a827999);  ROUND(c,d,a,b,11,9,G,0x5a827999); ROUND(b,c,d,a,15,13,G,0x5a827999);
    ROUND(a,b,c,d,0,3,H,0x6ed9eba1); ROUND(d,a,b,c,8,9,H,0x6ed9eba1); ROUND(c,d,a,b,4,11,H,0x6ed9eba1); ROUND(b,c,d,a,12,15,H,0x6ed9eba1);
    ROUND(a,b,c,d,2,3,H,0x6ed9eba1); ROUND(d,a,b,c,10,9,H,0x6ed9eba1); ROUND(c,d,a,b,6,11,H,0x6ed9eba1); ROUND(b,c,d,a,14,15,H,0x6ed9eba1);
    ROUND(a,b,c,d,1,3,H,0x6ed9eba1); ROUND(d,a,b,c,9,9,H,0x6ed9eba1); ROUND(c,d,a,b,5,11,H,0x6ed9eba1); ROUND(b,c,d,a,13,15,H,0x6ed9eba1);
    ROUND(a,b,c,d,3,3,H,0x6ed9eba1); ROUND(d,a,b,c,11,9,H,0x6ed9eba1); ROUND(c,d,a,b,7,11,H,0x6ed9eba1); ROUND(b,c,d,a,15,15,H,0x6ed9eba1);
#undef ROUND
#undef H
#undef G
#undef F
    a+=0x67452301; b+=0xefcdab89; c+=0x98badcfe; d+=0x10325476;
    uint32_t hv[4]={a,b,c,d};
    for(int i=0;i<4;i++) { out[i*4]=hv[i]&0xff; out[i*4+1]=(hv[i]>>8)&0xff; out[i*4+2]=(hv[i]>>16)&0xff; out[i*4+3]=(hv[i]>>24)&0xff; }
}

__device__ void compute_hash(const char *pwd, int len, uint8_t *out) {
    uint8_t type = d_hash_type;
    if(type==0){
        if(d_hash_len==20) type=2; else type=1; // default based on length
    }
    if(type==3){
        uint8_t buf[MAX_PWD_BYTES*2];
        int ulen=0;
        for(int i=0;i<len;i++) { buf[ulen++]=(uint8_t)pwd[i]; buf[ulen++]=0; }
        md4(buf, ulen, out);
    } else if(type==2){
        sha1(pwd, len, out);
    } else {
        md5(pwd, len, out);
    }
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

__global__ void crack_kernel(uint64_t start, uint64_t total, char *results,
                             int max_results, int *found_count) {
    uint64_t idx = start + threadIdx.x + blockIdx.x * (uint64_t)blockDim.x;
    uint64_t step = gridDim.x * (uint64_t)blockDim.x;
    uint8_t pwd[MAX_PWD_BYTES];
    uint8_t idx_buf[MAX_MASK_LEN];
    uint8_t digest[20];

    uint64_t end = start + total;
    while (idx < end) {
        uint64_t val = idx;
        for (int pos = d_pwd_len - 1; pos >= 0; --pos) {
            int set = d_pos_charset[pos];
            int len = d_charset_lens[set];
            idx_buf[pos] = val % len;
            val /= len;
        }
        int out_len = 0;
        for (int pos = 0; pos < d_pwd_len; ++pos) {
            int set = d_pos_charset[pos];
            int ci = idx_buf[pos];
            int clen = d_charset_charlen[set][ci];
            for (int j=0; j<clen; ++j)
                pwd[out_len++] = d_charset_bytes[set][ci][j];
        }
        compute_hash((char*)pwd, out_len, digest);
        if (check_hash(digest)) {
            int slot = atomicAdd(found_count, 1);
            if (slot < max_results) {
                int off = slot*MAX_PWD_BYTES;
                for(int i=0;i<out_len;i++) results[off+i] = pwd[i];
                results[off+out_len] = '\0';
            }
        }
        idx += step;
    }
}

extern "C" void load_darkling_data(const uint8_t **charset_bytes,
                                   const uint8_t **charset_lens,
                                   const int *charset_sizes,
                                   const uint8_t *pos_map, int pwd_len,
                                   const uint8_t *hashes, int num_hashes, int hash_len,
                                   uint8_t hash_type)
{
    cudaMemcpyToSymbol(d_pwd_len, &pwd_len, sizeof(int));
    cudaMemcpyToSymbol(d_hash_len, &hash_len, sizeof(int));
    cudaMemcpyToSymbol(d_hash_type, &hash_type, sizeof(uint8_t));
    cudaMemcpyToSymbol(d_num_hashes, &num_hashes, sizeof(int));
    cudaMemcpyToSymbol(d_pos_charset, pos_map, pwd_len);
    for(int i=0;i<MAX_CUSTOM_SETS; i++) {
        cudaMemcpyToSymbol(d_charset_lens, &charset_sizes[i], sizeof(int), i*sizeof(int));
        if(charset_sizes[i] > 0) {
            cudaMemcpyToSymbol(d_charset_bytes, charset_bytes[i], charset_sizes[i]*MAX_UTF8_BYTES,
                               i*MAX_CHARSET_CHARS*MAX_UTF8_BYTES);
            cudaMemcpyToSymbol(d_charset_charlen, charset_lens[i], charset_sizes[i],
                               i*MAX_CHARSET_CHARS);
        }
    }
    cudaMemcpyToSymbol(d_hashes, hashes, num_hashes * hash_len);
}

extern "C" void launch_darkling_kernel(uint64_t start, uint64_t end,
                                        char *d_results, int max_results, int *d_count,
                                        dim3 grid, dim3 block)
{
    uint64_t total = end - start;
    crack_kernel<<<grid, block>>>(start, total, d_results, max_results, d_count);
}

extern "C" void launch_darkling(const uint8_t **charset_bytes,
                                 const uint8_t **charset_lens,
                                 const int *charset_sizes,
                                 const uint8_t *pos_map, int pwd_len,
                                 const uint8_t *hashes, int num_hashes, int hash_len,
                                 uint8_t hash_type,
                                 uint64_t start, uint64_t end,
                                 char *d_results, int max_results, int *d_count,
                                 dim3 grid, dim3 block)
{
    load_darkling_data(charset_bytes, charset_lens, charset_sizes, pos_map,
                       pwd_len, hashes, num_hashes, hash_len, hash_type);
    launch_darkling_kernel(start, end, d_results, max_results, d_count, grid, block);
}
