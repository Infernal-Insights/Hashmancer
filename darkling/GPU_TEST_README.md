# GPU Testing Instructions for Darkling CLI Integration

## Overview
This test suite validates the Darkling-Hashmancer CLI integration on actual NVIDIA GPU hardware.

## Prerequisites
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed (`nvidia-smi` working)
- CUDA Toolkit installed (`nvcc` available)
- Git repository with the updated Darkling code

## Quick Start

1. **Copy the code to your GPU system:**
   ```bash
   # Transfer the entire darkling directory to your GPU system
   scp -r /path/to/hashmancer/darkling user@gpu-system:/path/to/destination/
   ```

2. **Run the automated test suite:**
   ```bash
   cd darkling
   ./gpu_test_suite.sh
   ```

## What the Test Suite Validates

### 🏗️ Build Testing
- Clean CUDA build with all new CLI integration code
- CMake configuration with CUDA support
- PTX rule compilation
- All dependencies and linking

### 🖥️ CLI Interface Testing  
- Argument parsing compatibility with Hashmancer worker commands
- Support for `--rules`, `--shard`, `--outfile-format`, `--status`, etc.
- Help system functionality
- Error handling for invalid arguments

### 📜 Rule Processing Testing
- Hashcat rule file parsing and conversion
- Built-in rule mapping to PTX optimized versions
- Rule execution on GPU hardware
- Performance of rule-based attacks

### 🎯 Attack Mode Testing
- Dictionary attacks (`-a 0`)
- Mask attacks (`-a 3`) 
- Rule-based dictionary attacks
- Worker-compatible command execution

### 📊 Status & Output Testing
- Status reporting (`--status`)
- JSON status output (`--status-json`)
- Output file generation (`--outfile`)
- Multiple output formats (`--outfile-format`)

### 🚀 Performance Testing
- GPU memory usage validation
- Hash rate measurements
- Rule execution performance
- Overall system stability

## Expected Results

### ✅ Success Indicators
- All builds complete without errors
- Commands execute without crashes
- GPU memory is properly allocated/freed
- Status outputs are generated correctly
- Hash processing proceeds as expected

### ⚠️ Potential Issues to Watch For
- CUDA out-of-memory errors (adjust workload if needed)
- PTX compilation failures (check CUDA architecture support)
- Rule parsing errors (verify rule syntax)
- Performance regressions compared to original Darkling

## Manual Testing Commands

After the automated tests, try these manual tests:

### Basic Dictionary Attack
```bash
./main -m 0 /path/to/hashes.txt -a 0 /path/to/wordlist.txt --quiet
```

### Rule-Based Attack (Hashmancer Worker Style)
```bash
./main -m 0 hashes.txt --shard wordlist.txt --rules best64.rule -d 1 --quiet --status --status-json --outfile cracked.txt --outfile-format 2
```

### Mask Attack with Status
```bash
./main -m 0 hashes.txt -a 3 ?d?d?d?d?d?d --status --status-timer 5 --quiet
```

### Performance Comparison
```bash
# Test with new CLI integration
time ./main -m 0 hashes.txt -a 0 wordlist.txt -r rules.txt --quiet

# Compare against any previous Darkling version if available
```

## Troubleshooting

### Build Issues
- **CUDA not found**: Ensure CUDA toolkit is installed and `nvcc` is in PATH
- **Architecture mismatch**: Update CMakeLists.txt with your GPU's compute capability
- **Memory errors**: Reduce batch sizes in CUDA kernels

### Runtime Issues  
- **GPU out of memory**: Use `-w 1` for lower workload profile
- **Segmentation faults**: Check rule file syntax and hash file format
- **Performance issues**: Verify PTX rules are compiling correctly

### Integration Issues
- **CLI parsing errors**: Compare with hashcat command syntax
- **Rule execution failures**: Test individual rules with simple wordlists
- **Output format problems**: Verify file permissions and disk space

## Reporting Results

When reporting test results, please include:

1. **System Information:**
   - GPU model and memory
   - CUDA version
   - Driver version
   - OS details

2. **Test Results:**
   - Complete output from `./gpu_test_suite.sh`
   - Any error messages or warnings
   - Performance metrics if available

3. **Manual Test Results:**
   - Success/failure of manual commands
   - Any unexpected behavior
   - Performance comparisons

## Next Steps After Testing

If tests pass successfully:
- ✅ Integration is complete and ready for production
- 🚀 Deploy to Hashmancer worker environments
- 📈 Monitor performance in production workloads

If tests reveal issues:
- 🐛 Document specific failures and error messages
- 🔧 Identify whether issues are build, runtime, or integration-related  
- 📋 Prioritize fixes based on severity and impact

## Files Modified in Integration

For reference, the key files changed in this integration:
- `src/main.cu` - Complete CLI interface overhaul
- `src/hashcat_rule_parser.cpp` - New hashcat rule parser
- `include/darkling_rule_manager.h` - Extended rule manager interface
- `tests/test_cli_integration.cpp` - Integration test suite
- `CMakeLists.txt` - Build configuration updates

The integration maintains full backward compatibility while adding comprehensive Hashmancer worker support.