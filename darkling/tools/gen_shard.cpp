#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <input.txt> <output.bin>\n", argv[0]);
    return 1;
  }
  const char* in_path = argv[1];
  const char* out_path = argv[2];
  FILE* in = fopen(in_path, "rb");
  if (!in) { perror("open input"); return 1; }
  std::vector<uint32_t> offsets; offsets.push_back(0);
  std::vector<char> words;
  char buf[4096];
  uint32_t off = 0;
  while (fgets(buf, sizeof(buf), in)) {
    size_t len = strlen(buf);
    words.insert(words.end(), buf, buf + len);
    off += (uint32_t)len;
    offsets.push_back(off);
  }
  fclose(in);
  uint32_t count = offsets.size() - 1; // sentinel included
  FILE* out = fopen(out_path, "wb");
  if (!out) { perror("open output"); return 1; }
  fwrite(&count, sizeof(uint32_t), 1, out);
  fwrite(offsets.data(), sizeof(uint32_t), offsets.size(), out);
  if (!words.empty()) fwrite(words.data(), 1, words.size(), out);
  fclose(out);
  return 0;
}

