/* OpenCL port of darkling_engine.cu */
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define MAX_MASK_LEN 32
#define MAX_UTF8_BYTES 4
#define MAX_CUSTOM_SETS 16
#define MAX_CHARSET_CHARS 256
#define MAX_PWD_BYTES (MAX_MASK_LEN * MAX_UTF8_BYTES)
#define MAX_HASHES 1024

inline uint rotl32(uint x, uint n) { return (x<<n) | (x>>(32-n)); }

void md5(const __private char *msg, int len, __global uchar *out) {
    const uint init[4] = {0x67452301,0xefcdab89,0x98badcfe,0x10325476};
    uint a=init[0], b=init[1], c=init[2], d=init[3];
    uchar buffer[64];
    for(int i=0;i<64;i++) buffer[i]=0;
    for(int i=0;i<len;i++) buffer[i]=(uchar)msg[i];
    buffer[len]=0x80;
    ulong bits=(ulong)len*8;
    for(int i=0;i<8;i++) buffer[56+i]=(bits>>(8*i))&0xff;
    const uint k[64]={
        0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
        0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
        0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
        0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
        0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
        0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
        0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
        0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391};
    const uint r[64]={7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
        5,9,14,20,5,9,14,20,5,9,14,20,5,9,14,20,
        4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
        6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21};
    uint w[16];
    for(int i=0;i<16;i++) w[i]=(uint)buffer[i*4]|((uint)buffer[i*4+1]<<8)|((uint)buffer[i*4+2]<<16)|((uint)buffer[i*4+3]<<24);
    uint aa=a,bb=b,cc=c,dd=d;
    for(int i=0;i<64;i++){
        uint f,g;
        if(i<16){f=(b&c)|(~b&d);g=i;} else if(i<32){f=(d&b)|(~d&c);g=(5*i+1)&15;} else if(i<48){f=b^c^d;g=(3*i+5)&15;} else {f=c^(b|~d);g=(7*i)&15;}
        uint temp=d; d=c; c=b; uint t=a+f+k[i]+w[g]; b+=rotl32(t,r[i]); a=temp;
    }
    a+=aa; b+=bb; c+=cc; d+=dd;
    uint out32[4]={a,b,c,d};
    for(int i=0;i<4;i++){ out[i*4]=(out32[i])&0xff; out[i*4+1]=(out32[i]>>8)&0xff; out[i*4+2]=(out32[i]>>16)&0xff; out[i*4+3]=(out32[i]>>24)&0xff; }
}

__kernel void crack_kernel(
    ulong start,
    ulong total,
    __global const uchar *pos_charset,
    __global const uchar *charset_bytes,
    __global const uchar *charset_len,
    __global const int *charset_size,
    __global const uchar *hashes,
    int num_hashes,
    int hash_len,
    int pwd_len,
    __global char *results,
    int max_results,
    __global int *count)
{
    ulong idx=start+get_global_id(0);
    ulong step=get_global_size(0);
    uchar pwd[MAX_PWD_BYTES];
    uchar idx_buf[MAX_MASK_LEN];
    uchar digest[20];
    ulong end=start+total;
    while(idx<end){
        ulong val=idx;
        for(int pos=pwd_len-1;pos>=0;--pos){
            int set=pos_charset[pos];
            int len=charset_size[set];
            idx_buf[pos]=val%len;
            val/=len;
        }
        int out_len=0;
        for(int pos=0;pos<pwd_len;++pos){
            int set=pos_charset[pos];
            int ci=idx_buf[pos];
            int clen=charset_len[set*MAX_CHARSET_CHARS+ci];
            for(int j=0;j<clen;j++) pwd[out_len++]=charset_bytes[(set*MAX_CHARSET_CHARS+ci)*MAX_UTF8_BYTES+j];
        }
        md5((const char*)pwd,out_len,digest);
        for(int h=0;h<num_hashes;++h){
            int match=1;
            for(int i=0;i<hash_len;i++){ if(digest[i]!=hashes[h*hash_len+i]){match=0;break;} }
            if(match){
                int slot=atomic_inc(count);
                if(slot<max_results){
                    int off=slot*MAX_PWD_BYTES;
                    for(int i=0;i<out_len;i++) results[off+i]=pwd[i];
                    results[off+out_len]='\0';
                }
            }
        }
        idx+=step;
    }
}
