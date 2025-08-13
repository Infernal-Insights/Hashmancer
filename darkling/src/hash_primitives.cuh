#pragma once
#include <stdint.h>

__device__ inline uint32_t rotl(uint32_t x, uint32_t n) { return (x << n) | (x >> (32 - n)); }

__device__ void md5_hash(const uint8_t* data, uint32_t len, uint32_t* out4) {
  const uint32_t init_a = 0x67452301u;
  const uint32_t init_b = 0xefcdab89u;
  const uint32_t init_c = 0x98badcfeu;
  const uint32_t init_d = 0x10325476u;
  uint32_t a = init_a, b = init_b, c = init_c, d = init_d;
  uint8_t block[64];
  for (uint32_t i = 0; i < 64; ++i) block[i] = 0;
  for (uint32_t i = 0; i < len; ++i) block[i] = data[i];
  block[len] = 0x80u;
  uint64_t bits = (uint64_t)len * 8u;
  for (int i = 0; i < 8; ++i) block[56+i] = (bits >> (8*i)) & 0xffu;
  uint32_t* x = (uint32_t*)block;
#define F(x,y,z) ((x & y) | (~x & z))
#define G(x,y,z) ((x & z) | (y & ~z))
#define H(x,y,z) (x ^ y ^ z)
#define I(x,y,z) (y ^ (x | ~z))
#define OP(a,b,c,d,k,s,t,fn) a = b + rotl(a + fn(b,c,d) + x[k] + t, s);
  OP(a,b,c,d,0,7,0xd76aa478u,F);  OP(d,a,b,c,1,12,0xe8c7b756u,F);
  OP(c,d,a,b,2,17,0x242070dbu,F); OP(b,c,d,a,3,22,0xc1bdceeau,F);
  OP(a,b,c,d,4,7,0xf57c0fafu,F);  OP(d,a,b,c,5,12,0x4787c62au,F);
  OP(c,d,a,b,6,17,0xa8304613u,F); OP(b,c,d,a,7,22,0xfd469501u,F);
  OP(a,b,c,d,8,7,0x698098d8u,F);  OP(d,a,b,c,9,12,0x8b44f7afu,F);
  OP(c,d,a,b,10,17,0xffff5bb1u,F);OP(b,c,d,a,11,22,0x895cd7beu,F);
  OP(a,b,c,d,12,7,0x6b901122u,F); OP(d,a,b,c,13,12,0xfd987193u,F);
  OP(c,d,a,b,14,17,0xa679438eu,F);OP(b,c,d,a,15,22,0x49b40821u,F);
  OP(a,b,c,d,1,5,0xf61e2562u,G);  OP(d,a,b,c,6,9,0xc040b340u,G);
  OP(c,d,a,b,11,14,0x265e5a51u,G);OP(b,c,d,a,0,20,0xe9b6c7aau,G);
  OP(a,b,c,d,5,5,0xd62f105du,G);  OP(d,a,b,c,10,9,0x02441453u,G);
  OP(c,d,a,b,15,14,0xd8a1e681u,G);OP(b,c,d,a,4,20,0xe7d3fbc8u,G);
  OP(a,b,c,d,9,5,0x21e1cde6u,G);  OP(d,a,b,c,14,9,0xc33707d6u,G);
  OP(c,d,a,b,3,14,0xf4d50d87u,G); OP(b,c,d,a,8,20,0x455a14edu,G);
  OP(a,b,c,d,13,5,0xa9e3e905u,G); OP(d,a,b,c,2,9,0xfcefa3f8u,G);
  OP(c,d,a,b,7,14,0x676f02d9u,G); OP(b,c,d,a,12,20,0x8d2a4c8au,G);
  OP(a,b,c,d,5,4,0xfffa3942u,H);  OP(d,a,b,c,8,11,0x8771f681u,H);
  OP(c,d,a,b,11,16,0x6d9d6122u,H);OP(b,c,d,a,14,23,0xfde5380cu,H);
  OP(a,b,c,d,1,4,0xa4beea44u,H);  OP(d,a,b,c,4,11,0x4bdecfa9u,H);
  OP(c,d,a,b,7,16,0xf6bb4b60u,H); OP(b,c,d,a,10,23,0xbebfbc70u,H);
  OP(a,b,c,d,13,4,0x289b7ec6u,H); OP(d,a,b,c,0,11,0xeaa127fau,H);
  OP(c,d,a,b,3,16,0xd4ef3085u,H); OP(b,c,d,a,6,23,0x04881d05u,H);
  OP(a,b,c,d,9,4,0xd9d4d039u,H);  OP(d,a,b,c,12,11,0xe6db99e5u,H);
  OP(c,d,a,b,15,16,0x1fa27cf8u,H);OP(b,c,d,a,2,23,0xc4ac5665u,H);
  OP(a,b,c,d,0,6,0xf4292244u,I);  OP(d,a,b,c,7,10,0x432aff97u,I);
  OP(c,d,a,b,14,15,0xab9423a7u,I);OP(b,c,d,a,5,21,0xfc93a039u,I);
  OP(a,b,c,d,12,6,0x655b59c3u,I); OP(d,a,b,c,3,10,0x8f0ccc92u,I);
  OP(c,d,a,b,10,15,0xffeff47du,I);OP(b,c,d,a,1,21,0x85845dd1u,I);
  OP(a,b,c,d,8,6,0x6fa87e4fu,I);  OP(d,a,b,c,15,10,0xfe2ce6e0u,I);
  OP(c,d,a,b,6,15,0xa3014314u,I);OP(b,c,d,a,13,21,0x4e0811a1u,I);
  OP(a,b,c,d,4,6,0xf7537e82u,I);  OP(d,a,b,c,11,10,0xbd3af235u,I);
  OP(c,d,a,b,2,15,0x2ad7d2bbu,I);OP(b,c,d,a,9,21,0xeb86d391u,I);
  out4[0] = a + init_a;
  out4[1] = b + init_b;
  out4[2] = c + init_c;
  out4[3] = d + init_d;
#undef F
#undef G
#undef H
#undef I
#undef OP
}

__device__ inline void sha1_hash(const uint8_t*, uint32_t, uint32_t*) {}
__device__ inline void ntlm_hash(const uint8_t*, uint32_t, uint32_t*) {}
__device__ inline bool check_hash(const uint32_t*) { return false; }
