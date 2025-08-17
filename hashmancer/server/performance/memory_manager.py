"""Memory-mapped file processing and optimized memory management."""

import os
import mmap
import time
import logging
from typing import Iterator, List, Optional, Dict, Any, BinaryIO
from pathlib import Path
from contextlib import contextmanager
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FileStats:
    """Statistics for memory-mapped files."""
    file_path: str
    file_size: int
    line_count: int = 0
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0


class MemoryMappedWordlist:
    """High-performance memory-mapped wordlist reader."""
    
    def __init__(self, file_path: str, cache_stats: bool = True):
        self.file_path = Path(file_path)
        self.cache_stats = cache_stats
        self._mmap: Optional[mmap.mmap] = None
        self._file: Optional[BinaryIO] = None
        self._stats: Optional[FileStats] = None
        self._line_offsets: Optional[List[int]] = None
        self._lock = threading.RLock()
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Wordlist file not found: {file_path}")
        
        self._initialize()
    
    def _initialize(self):
        """Initialize memory mapping and statistics."""
        with self._lock:
            file_size = self.file_path.stat().st_size
            
            if self.cache_stats:
                self._stats = FileStats(
                    file_path=str(self.file_path),
                    file_size=file_size,
                    created_at=time.time()
                )
            
            self._file = open(self.file_path, 'rb')
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Pre-compute line offsets for random access
            if file_size < 100 * 1024 * 1024:  # Only for files < 100MB
                self._compute_line_offsets()
            
            logger.info(f"Memory-mapped wordlist: {self.file_path} ({file_size:,} bytes)")
    
    def _compute_line_offsets(self):
        """Pre-compute line offsets for O(1) line access."""
        self._line_offsets = [0]
        
        pos = 0
        while pos < len(self._mmap):
            # Find next newline
            newline_pos = self._mmap.find(b'\n', pos)
            if newline_pos == -1:
                break
            self._line_offsets.append(newline_pos + 1)
            pos = newline_pos + 1
        
        if self._stats:
            self._stats.line_count = len(self._line_offsets) - 1
        
        logger.debug(f"Computed {len(self._line_offsets)} line offsets for {self.file_path}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Close memory-mapped file."""
        with self._lock:
            if self._mmap:
                self._mmap.close()
                self._mmap = None
            
            if self._file:
                self._file.close()
                self._file = None
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over lines in the wordlist."""
        return self.iter_lines()
    
    def iter_lines(self, start_line: int = 0, max_lines: Optional[int] = None) -> Iterator[str]:
        """Iterate over lines with optional start and limit."""
        if not self._mmap:
            raise RuntimeError("Memory-mapped file is closed")
        
        with self._lock:
            if self._stats:
                self._stats.last_accessed = time.time()
                self._stats.access_count += 1
            
            lines_read = 0
            
            if self._line_offsets and start_line > 0:
                # Jump to specific line using pre-computed offsets
                if start_line < len(self._line_offsets):
                    self._mmap.seek(self._line_offsets[start_line])
                else:
                    return  # Start line beyond file
            else:
                self._mmap.seek(0)
                # Skip lines if needed without offsets
                for _ in range(start_line):
                    try:
                        line = self._mmap.readline()
                        if not line:
                            return
                    except ValueError:
                        return
            
            while True:
                if max_lines is not None and lines_read >= max_lines:
                    break
                
                try:
                    line = self._mmap.readline()
                    if not line:
                        break
                    
                    # Decode and strip line
                    decoded_line = line.decode('utf-8', errors='ignore').rstrip('\r\n')
                    if decoded_line:  # Skip empty lines
                        yield decoded_line
                        lines_read += 1
                
                except (ValueError, UnicodeDecodeError) as e:
                    logger.warning(f"Error reading line from {self.file_path}: {e}")
                    continue
    
    def get_line(self, line_number: int) -> Optional[str]:
        """Get specific line by number (0-indexed)."""
        if not self._line_offsets:
            # Fallback to sequential reading
            for i, line in enumerate(self.iter_lines()):
                if i == line_number:
                    return line
            return None
        
        if line_number >= len(self._line_offsets) - 1:
            return None
        
        with self._lock:
            start_offset = self._line_offsets[line_number]
            end_offset = self._line_offsets[line_number + 1] - 1  # Exclude newline
            
            line_bytes = self._mmap[start_offset:end_offset]
            return line_bytes.decode('utf-8', errors='ignore').rstrip('\r\n')
    
    def get_lines_range(self, start: int, end: int) -> List[str]:
        """Get range of lines efficiently."""
        return list(self.iter_lines(start_line=start, max_lines=end - start))
    
    def search(self, pattern: str, case_sensitive: bool = True, max_results: int = 1000) -> List[tuple[int, str]]:
        """Search for pattern in wordlist."""
        results = []
        search_pattern = pattern if case_sensitive else pattern.lower()
        
        for line_no, line in enumerate(self.iter_lines()):
            if len(results) >= max_results:
                break
            
            search_line = line if case_sensitive else line.lower()
            if search_pattern in search_line:
                results.append((line_no, line))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get wordlist statistics."""
        if not self._stats:
            return {'error': 'Statistics not enabled'}
        
        stats = {
            'file_path': self._stats.file_path,
            'file_size': self._stats.file_size,
            'file_size_mb': self._stats.file_size / (1024 * 1024),
            'line_count': self._stats.line_count,
            'created_at': self._stats.created_at,
            'last_accessed': self._stats.last_accessed,
            'access_count': self._stats.access_count,
            'has_line_index': self._line_offsets is not None,
            'line_index_size': len(self._line_offsets) if self._line_offsets else 0
        }
        
        if self._stats.last_accessed > 0:
            stats['last_access_ago'] = time.time() - self._stats.last_accessed
        
        return stats


class WordlistCache:
    """LRU cache for memory-mapped wordlists."""
    
    def __init__(self, max_files: int = 10, max_memory_mb: int = 500):
        self.max_files = max_files
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: Dict[str, MemoryMappedWordlist] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()
        self._current_memory = 0
    
    def get_wordlist(self, file_path: str) -> MemoryMappedWordlist:
        """Get wordlist from cache or load it."""
        file_path = str(Path(file_path).resolve())
        
        with self._lock:
            # Check if already in cache
            if file_path in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(file_path)
                self._access_order.append(file_path)
                return self._cache[file_path]
            
            # Load new wordlist
            wordlist = MemoryMappedWordlist(file_path)
            file_size = Path(file_path).stat().st_size
            
            # Make room if necessary
            while (len(self._cache) >= self.max_files or 
                   self._current_memory + file_size > self.max_memory_bytes):
                if not self._access_order:
                    break
                self._evict_oldest()
            
            # Add to cache
            self._cache[file_path] = wordlist
            self._access_order.append(file_path)
            self._current_memory += file_size
            
            logger.info(f"Cached wordlist: {file_path} (cache size: {len(self._cache)})")
            return wordlist
    
    def _evict_oldest(self):
        """Evict least recently used wordlist."""
        if not self._access_order:
            return
        
        oldest_path = self._access_order.pop(0)
        if oldest_path in self._cache:
            wordlist = self._cache.pop(oldest_path)
            file_size = Path(oldest_path).stat().st_size
            self._current_memory -= file_size
            
            try:
                wordlist.close()
            except Exception as e:
                logger.warning(f"Error closing wordlist {oldest_path}: {e}")
            
            logger.debug(f"Evicted wordlist from cache: {oldest_path}")
    
    def clear(self):
        """Clear all cached wordlists."""
        with self._lock:
            for wordlist in self._cache.values():
                try:
                    wordlist.close()
                except Exception as e:
                    logger.warning(f"Error closing wordlist during clear: {e}")
            
            self._cache.clear()
            self._access_order.clear()
            self._current_memory = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'cached_files': len(self._cache),
                'max_files': self.max_files,
                'current_memory_mb': self._current_memory / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'memory_utilization': self._current_memory / self.max_memory_bytes if self.max_memory_bytes > 0 else 0,
                'cached_paths': list(self._cache.keys())
            }


# Global wordlist cache
_wordlist_cache: Optional[WordlistCache] = None


def get_wordlist_cache() -> WordlistCache:
    """Get global wordlist cache."""
    global _wordlist_cache
    if _wordlist_cache is None:
        _wordlist_cache = WordlistCache()
    return _wordlist_cache


@contextmanager
def optimized_file_reader(file_path: str, use_mmap: bool = True):
    """Context manager for optimized file reading."""
    if use_mmap:
        try:
            wordlist = get_wordlist_cache().get_wordlist(file_path)
            yield wordlist
        except Exception as e:
            logger.warning(f"Memory mapping failed for {file_path}, falling back to regular file: {e}")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                yield f
    else:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            yield f


class ChunkedFileProcessor:
    """Process large files in chunks to control memory usage."""
    
    def __init__(self, chunk_size: int = 64 * 1024):  # 64KB chunks
        self.chunk_size = chunk_size
    
    def process_file(self, file_path: str, processor_func, **kwargs):
        """Process file in chunks with custom processor function."""
        results = []
        
        with open(file_path, 'rb') as f:
            buffer = b''
            
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    # Process remaining buffer
                    if buffer:
                        result = processor_func(buffer, is_final=True, **kwargs)
                        if result is not None:
                            results.append(result)
                    break
                
                buffer += chunk
                
                # Process complete lines in buffer
                lines = buffer.split(b'\n')
                buffer = lines[-1]  # Keep incomplete line in buffer
                
                for line in lines[:-1]:
                    if line:  # Skip empty lines
                        try:
                            decoded_line = line.decode('utf-8', errors='ignore').rstrip('\r')
                            result = processor_func(decoded_line, is_final=False, **kwargs)
                            if result is not None:
                                results.append(result)
                        except Exception as e:
                            logger.warning(f"Error processing line: {e}")
        
        return results
    
    def count_lines(self, file_path: str) -> int:
        """Efficiently count lines in a file."""
        line_count = 0
        
        def line_counter(line_data, is_final=False):
            nonlocal line_count
            if isinstance(line_data, bytes):
                line_count += line_data.count(b'\n')
            else:
                line_count += 1
            return None
        
        self.process_file(file_path, line_counter)
        return line_count
    
    def extract_patterns(self, file_path: str, pattern_func) -> List[Any]:
        """Extract patterns from file using custom pattern function."""
        patterns = []
        
        def pattern_extractor(line, is_final=False):
            if not is_final:
                pattern = pattern_func(line)
                if pattern:
                    return pattern
            return None
        
        return self.process_file(file_path, pattern_extractor)


def get_file_stats(file_path: str) -> Dict[str, Any]:
    """Get comprehensive file statistics."""
    path = Path(file_path)
    
    if not path.exists():
        return {'error': 'File not found'}
    
    stat = path.stat()
    
    # Quick line count for small files
    line_count = 0
    if stat.st_size < 50 * 1024 * 1024:  # < 50MB
        try:
            with open(path, 'rb') as f:
                line_count = sum(1 for _ in f)
        except Exception:
            line_count = -1
    
    return {
        'file_path': str(path),
        'file_size': stat.st_size,
        'file_size_mb': stat.st_size / (1024 * 1024),
        'line_count': line_count,
        'created_at': stat.st_ctime,
        'modified_at': stat.st_mtime,
        'is_readable': os.access(path, os.R_OK),
        'suggested_mmap': stat.st_size > 1024 * 1024  # Suggest mmap for files > 1MB
    }