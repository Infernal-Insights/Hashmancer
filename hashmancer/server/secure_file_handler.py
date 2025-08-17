"""
Secure File Handling System
Provides secure file operations with proper validation, sandboxing, and edge case handling
"""

import os
import shutil
import tempfile
import hashlib
import mimetypes
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, BinaryIO, TextIO
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
import magic  # python-magic for file type detection
from contextlib import contextmanager
import fcntl  # File locking on Unix systems
import stat
import pwd
import grp

logger = logging.getLogger(__name__)


class FileAccessMode(Enum):
    """File access modes."""
    READ_ONLY = "r"
    WRITE_ONLY = "w"
    READ_WRITE = "r+"
    APPEND = "a"
    BINARY_READ = "rb"
    BINARY_WRITE = "wb"
    BINARY_READ_WRITE = "rb+"


class FileValidationError(Exception):
    """Exception raised for file validation errors."""
    pass


class FileSecurityError(Exception):
    """Exception raised for file security violations."""
    pass


@dataclass
class FileInfo:
    """File information structure."""
    path: Path
    size: int
    mime_type: str
    file_type: str
    permissions: str
    owner: str
    group: str
    created_time: float
    modified_time: float
    accessed_time: float
    is_executable: bool
    is_symlink: bool
    is_hidden: bool
    checksum_md5: str
    checksum_sha256: str


class SecureFileValidator:
    """Validates files for security and integrity."""
    
    def __init__(self):
        # Allowed file extensions
        self.allowed_extensions = {
            '.txt', '.log', '.json', '.csv', '.xml', '.yaml', '.yml',
            '.py', '.js', '.html', '.css', '.md', '.rst',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.zip', '.tar', '.gz', '.bz2',
            '.rule', '.rules'  # Hashmancer-specific
        }
        
        # Dangerous file extensions to always reject
        self.dangerous_extensions = {
            '.exe', '.bat', '.cmd', '.com', '.scr', '.pif',
            '.vbs', '.vbe', '.js', '.jse', '.wsf', '.wsh',
            '.msi', '.msp', '.dll', '.so', '.dylib',
            '.sh', '.bash', '.csh', '.fish', '.zsh'
        }
        
        # Allowed MIME types
        self.allowed_mime_types = {
            'text/plain', 'text/html', 'text/css', 'text/javascript',
            'text/csv', 'text/xml', 'text/yaml',
            'application/json', 'application/xml', 'application/yaml',
            'application/pdf', 'application/zip',
            'application/msword', 'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/svg+xml'
        }
        
        # Maximum file sizes (in bytes)
        self.max_file_sizes = {
            'default': 100 * 1024 * 1024,  # 100MB
            'image': 50 * 1024 * 1024,     # 50MB
            'text': 10 * 1024 * 1024,      # 10MB
            'document': 200 * 1024 * 1024,  # 200MB
        }
        
        # File type detection
        self.magic = magic.Magic(mime=True)
    
    def validate_filename(self, filename: str) -> bool:
        """Validate filename for security issues."""
        if not filename:
            raise FileValidationError("Filename cannot be empty")
        
        # Check for path traversal attempts
        if '..' in filename or '/' in filename or '\\' in filename:
            raise FileSecurityError("Path traversal attempt detected in filename")
        
        # Check for hidden files (unless explicitly allowed)
        if filename.startswith('.') and filename not in {'.htaccess', '.gitignore'}:
            raise FileSecurityError("Hidden files not allowed")
        
        # Check filename length
        if len(filename) > 255:
            raise FileValidationError("Filename too long")
        
        # Check for dangerous characters
        dangerous_chars = {'<', '>', ':', '"', '|', '?', '*', '\x00'}
        if any(char in filename for char in dangerous_chars):
            raise FileSecurityError("Dangerous characters in filename")
        
        # Check file extension
        extension = Path(filename).suffix.lower()
        if extension in self.dangerous_extensions:
            raise FileSecurityError(f"Dangerous file extension: {extension}")
        
        return True
    
    def validate_file_content(self, file_path: Path) -> FileInfo:
        """Validate file content and return file information."""
        if not file_path.exists():
            raise FileValidationError("File does not exist")
        
        if not file_path.is_file():
            raise FileValidationError("Path is not a regular file")
        
        # Get file stats
        stat_info = file_path.stat()
        
        # Check file size
        file_size = stat_info.st_size
        max_size = self._get_max_size_for_file(file_path)
        if file_size > max_size:
            raise FileValidationError(f"File too large: {file_size} bytes (max: {max_size})")
        
        # Detect MIME type
        try:
            mime_type = self.magic.from_file(str(file_path))
        except Exception as e:
            logger.warning(f"Could not detect MIME type for {file_path}: {e}")
            mime_type = "application/octet-stream"
        
        # Validate MIME type
        if mime_type not in self.allowed_mime_types and not mime_type.startswith('text/'):
            logger.warning(f"Potentially unsafe MIME type: {mime_type}")
        
        # Calculate checksums
        md5_hash = self._calculate_checksum(file_path, 'md5')
        sha256_hash = self._calculate_checksum(file_path, 'sha256')
        
        # Get file permissions and ownership
        permissions = oct(stat_info.st_mode)[-3:]
        
        try:
            owner = pwd.getpwuid(stat_info.st_uid).pw_name
        except KeyError:
            owner = str(stat_info.st_uid)
        
        try:
            group = grp.getgrgid(stat_info.st_gid).gr_name
        except KeyError:
            group = str(stat_info.st_gid)
        
        # Create file info
        file_info = FileInfo(
            path=file_path,
            size=file_size,
            mime_type=mime_type,
            file_type=self.magic.from_file(str(file_path)),
            permissions=permissions,
            owner=owner,
            group=group,
            created_time=stat_info.st_ctime,
            modified_time=stat_info.st_mtime,
            accessed_time=stat_info.st_atime,
            is_executable=bool(stat_info.st_mode & stat.S_IXUSR),
            is_symlink=file_path.is_symlink(),
            is_hidden=file_path.name.startswith('.'),
            checksum_md5=md5_hash,
            checksum_sha256=sha256_hash
        )
        
        return file_info
    
    def _get_max_size_for_file(self, file_path: Path) -> int:
        """Get maximum allowed size for file based on type."""
        extension = file_path.suffix.lower()
        mime_type = mimetypes.guess_type(str(file_path))[0] or ''
        
        if mime_type.startswith('image/'):
            return self.max_file_sizes['image']
        elif mime_type.startswith('text/') or extension in {'.txt', '.log', '.csv', '.json'}:
            return self.max_file_sizes['text']
        elif mime_type.startswith('application/') and any(doc in mime_type for doc in ['word', 'excel', 'powerpoint', 'pdf']):
            return self.max_file_sizes['document']
        else:
            return self.max_file_sizes['default']
    
    def _calculate_checksum(self, file_path: Path, algorithm: str) -> str:
        """Calculate file checksum."""
        if algorithm == 'md5':
            hash_obj = hashlib.md5()
        elif algorithm == 'sha256':
            hash_obj = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating {algorithm} checksum for {file_path}: {e}")
            return ""


class SecureFileManager:
    """Secure file manager with sandboxing and safe operations."""
    
    def __init__(self, base_directory: Union[str, Path], temp_directory: Optional[Union[str, Path]] = None):
        self.base_directory = Path(base_directory).resolve()
        self.temp_directory = Path(temp_directory) if temp_directory else Path(tempfile.gettempdir())
        self.validator = SecureFileValidator()
        self.lock = threading.RLock()
        
        # Ensure directories exist
        self.base_directory.mkdir(parents=True, exist_ok=True)
        self.temp_directory.mkdir(parents=True, exist_ok=True)
        
        # File operation tracking
        self.open_files: Dict[str, Any] = {}
        self.operation_stats = {
            'files_created': 0,
            'files_deleted': 0,
            'files_read': 0,
            'files_written': 0,
            'errors': 0
        }
    
    def _ensure_safe_path(self, file_path: Union[str, Path]) -> Path:
        """Ensure file path is within allowed directory."""
        path = Path(file_path).resolve()
        
        # Check if path is within base directory
        try:
            path.relative_to(self.base_directory)
        except ValueError:
            raise FileSecurityError(f"Path outside allowed directory: {path}")
        
        return path
    
    @contextmanager
    def open_file(self, file_path: Union[str, Path], mode: FileAccessMode = FileAccessMode.READ_ONLY,
                  encoding: str = 'utf-8', validate: bool = True):
        """Securely open file with proper cleanup and locking."""
        safe_path = self._ensure_safe_path(file_path)
        
        if validate:
            self.validator.validate_filename(safe_path.name)
        
        file_id = f"{safe_path}_{mode.value}_{threading.get_ident()}"
        
        try:
            with self.lock:
                if mode in [FileAccessMode.BINARY_READ, FileAccessMode.BINARY_WRITE, FileAccessMode.BINARY_READ_WRITE]:
                    file_obj = open(safe_path, mode.value)
                else:
                    file_obj = open(safe_path, mode.value, encoding=encoding)
                
                # Apply file locking
                if hasattr(fcntl, 'LOCK_EX'):  # Unix systems
                    if 'w' in mode.value or 'a' in mode.value or '+' in mode.value:
                        fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    else:
                        fcntl.flock(file_obj.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                
                self.open_files[file_id] = {
                    'file_obj': file_obj,
                    'path': safe_path,
                    'mode': mode,
                    'opened_at': time.time()
                }
                
                logger.debug(f"Opened file: {safe_path} (mode: {mode.value})")
                
                # Update statistics
                if 'r' in mode.value:
                    self.operation_stats['files_read'] += 1
                if 'w' in mode.value or 'a' in mode.value:
                    self.operation_stats['files_written'] += 1
                
                yield file_obj
                
        except Exception as e:
            self.operation_stats['errors'] += 1
            logger.error(f"Error opening file {safe_path}: {e}")
            raise
        finally:
            with self.lock:
                if file_id in self.open_files:
                    try:
                        file_obj = self.open_files[file_id]['file_obj']
                        if hasattr(fcntl, 'LOCK_UN'):
                            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
                        file_obj.close()
                        del self.open_files[file_id]
                        logger.debug(f"Closed file: {safe_path}")
                    except Exception as e:
                        logger.error(f"Error closing file {safe_path}: {e}")
    
    def read_file(self, file_path: Union[str, Path], encoding: str = 'utf-8', 
                  max_size: Optional[int] = None) -> str:
        """Safely read text file."""
        safe_path = self._ensure_safe_path(file_path)
        
        # Validate file
        file_info = self.validator.validate_file_content(safe_path)
        
        # Check size limit
        if max_size and file_info.size > max_size:
            raise FileValidationError(f"File too large: {file_info.size} bytes (max: {max_size})")
        
        with self.open_file(safe_path, FileAccessMode.READ_ONLY, encoding=encoding) as f:
            return f.read()
    
    def read_file_binary(self, file_path: Union[str, Path], max_size: Optional[int] = None) -> bytes:
        """Safely read binary file."""
        safe_path = self._ensure_safe_path(file_path)
        
        # Validate file
        file_info = self.validator.validate_file_content(safe_path)
        
        # Check size limit
        if max_size and file_info.size > max_size:
            raise FileValidationError(f"File too large: {file_info.size} bytes (max: {max_size})")
        
        with self.open_file(safe_path, FileAccessMode.BINARY_READ) as f:
            return f.read()
    
    def write_file(self, file_path: Union[str, Path], content: str, 
                   encoding: str = 'utf-8', create_dirs: bool = True) -> FileInfo:
        """Safely write text file."""
        safe_path = self._ensure_safe_path(file_path)
        
        # Validate filename
        self.validator.validate_filename(safe_path.name)
        
        # Create parent directories if needed
        if create_dirs:
            safe_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first, then move
        temp_file = self.temp_directory / f"tmp_{safe_path.name}_{int(time.time())}"
        
        try:
            with open(temp_file, 'w', encoding=encoding) as f:
                f.write(content)
            
            # Move to final location
            shutil.move(str(temp_file), str(safe_path))
            
            self.operation_stats['files_created'] += 1
            logger.info(f"Created file: {safe_path}")
            
            # Return file info
            return self.validator.validate_file_content(safe_path)
            
        except Exception as e:
            # Clean up temp file on error
            if temp_file.exists():
                temp_file.unlink()
            self.operation_stats['errors'] += 1
            raise
    
    def write_file_binary(self, file_path: Union[str, Path], content: bytes, 
                         create_dirs: bool = True) -> FileInfo:
        """Safely write binary file."""
        safe_path = self._ensure_safe_path(file_path)
        
        # Validate filename
        self.validator.validate_filename(safe_path.name)
        
        # Create parent directories if needed
        if create_dirs:
            safe_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first, then move
        temp_file = self.temp_directory / f"tmp_{safe_path.name}_{int(time.time())}"
        
        try:
            with open(temp_file, 'wb') as f:
                f.write(content)
            
            # Move to final location
            shutil.move(str(temp_file), str(safe_path))
            
            self.operation_stats['files_created'] += 1
            logger.info(f"Created binary file: {safe_path}")
            
            # Return file info
            return self.validator.validate_file_content(safe_path)
            
        except Exception as e:
            # Clean up temp file on error
            if temp_file.exists():
                temp_file.unlink()
            self.operation_stats['errors'] += 1
            raise
    
    def delete_file(self, file_path: Union[str, Path], secure_delete: bool = False) -> bool:
        """Safely delete file."""
        safe_path = self._ensure_safe_path(file_path)
        
        if not safe_path.exists():
            return False
        
        if not safe_path.is_file():
            raise FileSecurityError("Can only delete regular files")
        
        try:
            if secure_delete:
                self._secure_delete(safe_path)
            else:
                safe_path.unlink()
            
            self.operation_stats['files_deleted'] += 1
            logger.info(f"Deleted file: {safe_path}")
            return True
            
        except Exception as e:
            self.operation_stats['errors'] += 1
            logger.error(f"Error deleting file {safe_path}: {e}")
            raise
    
    def _secure_delete(self, file_path: Path):
        """Securely delete file by overwriting with random data."""
        file_size = file_path.stat().st_size
        
        # Overwrite with random data 3 times
        with open(file_path, 'r+b') as f:
            for _ in range(3):
                f.seek(0)
                f.write(os.urandom(file_size))
                f.flush()
                os.fsync(f.fileno())
        
        # Finally delete the file
        file_path.unlink()
    
    def copy_file(self, src_path: Union[str, Path], dst_path: Union[str, Path], 
                  verify_checksum: bool = True) -> FileInfo:
        """Safely copy file with verification."""
        safe_src = self._ensure_safe_path(src_path)
        safe_dst = self._ensure_safe_path(dst_path)
        
        # Validate source file
        src_info = self.validator.validate_file_content(safe_src)
        
        # Validate destination filename
        self.validator.validate_filename(safe_dst.name)
        
        # Create parent directories
        safe_dst.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Copy file
            shutil.copy2(str(safe_src), str(safe_dst))
            
            # Verify copy if requested
            if verify_checksum:
                dst_info = self.validator.validate_file_content(safe_dst)
                if src_info.checksum_sha256 != dst_info.checksum_sha256:
                    safe_dst.unlink()  # Remove corrupted copy
                    raise FileValidationError("File copy verification failed")
            
            logger.info(f"Copied file: {safe_src} -> {safe_dst}")
            return self.validator.validate_file_content(safe_dst)
            
        except Exception as e:
            self.operation_stats['errors'] += 1
            logger.error(f"Error copying file {safe_src} to {safe_dst}: {e}")
            raise
    
    def list_files(self, directory: Optional[Union[str, Path]] = None, 
                   pattern: str = "*", recursive: bool = False) -> List[FileInfo]:
        """List files in directory with optional filtering."""
        if directory is None:
            safe_dir = self.base_directory
        else:
            safe_dir = self._ensure_safe_path(directory)
        
        if not safe_dir.is_dir():
            raise FileValidationError("Path is not a directory")
        
        files = []
        
        try:
            if recursive:
                file_paths = safe_dir.rglob(pattern)
            else:
                file_paths = safe_dir.glob(pattern)
            
            for file_path in file_paths:
                if file_path.is_file():
                    try:
                        file_info = self.validator.validate_file_content(file_path)
                        files.append(file_info)
                    except Exception as e:
                        logger.warning(f"Error processing file {file_path}: {e}")
                        continue
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files in {safe_dir}: {e}")
            raise
    
    def get_file_info(self, file_path: Union[str, Path]) -> FileInfo:
        """Get detailed file information."""
        safe_path = self._ensure_safe_path(file_path)
        return self.validator.validate_file_content(safe_path)
    
    def verify_file_integrity(self, file_path: Union[str, Path], 
                             expected_checksum: str, algorithm: str = 'sha256') -> bool:
        """Verify file integrity using checksum."""
        safe_path = self._ensure_safe_path(file_path)
        
        if algorithm == 'sha256':
            actual_checksum = self.validator._calculate_checksum(safe_path, 'sha256')
        elif algorithm == 'md5':
            actual_checksum = self.validator._calculate_checksum(safe_path, 'md5')
        else:
            raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
        
        return actual_checksum.lower() == expected_checksum.lower()
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up old temporary files."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleaned_count = 0
        
        try:
            for temp_file in self.temp_directory.glob("tmp_*"):
                if temp_file.stat().st_mtime < cutoff_time:
                    temp_file.unlink()
                    cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} temporary files")
                
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get file operation statistics."""
        with self.lock:
            return {
                'operation_stats': self.operation_stats.copy(),
                'open_files_count': len(self.open_files),
                'base_directory': str(self.base_directory),
                'temp_directory': str(self.temp_directory),
                'open_files': [
                    {
                        'path': str(info['path']),
                        'mode': info['mode'].value,
                        'opened_at': info['opened_at']
                    }
                    for info in self.open_files.values()
                ]
            }


# Global file manager instances
_file_managers: Dict[str, SecureFileManager] = {}
_manager_lock = threading.Lock()


def get_file_manager(name: str = "default", base_directory: Optional[str] = None) -> SecureFileManager:
    """Get a named file manager instance."""
    global _file_managers
    
    if name not in _file_managers:
        with _manager_lock:
            if name not in _file_managers:
                if base_directory is None:
                    base_directory = f"/tmp/hashmancer_files_{name}"
                _file_managers[name] = SecureFileManager(base_directory)
    
    return _file_managers[name]


# Convenience functions
def safe_read_file(file_path: Union[str, Path], manager_name: str = "default") -> str:
    """Safely read a text file."""
    manager = get_file_manager(manager_name)
    return manager.read_file(file_path)


def safe_write_file(file_path: Union[str, Path], content: str, manager_name: str = "default") -> FileInfo:
    """Safely write a text file."""
    manager = get_file_manager(manager_name)
    return manager.write_file(file_path, content)


def safe_delete_file(file_path: Union[str, Path], manager_name: str = "default") -> bool:
    """Safely delete a file."""
    manager = get_file_manager(manager_name)
    return manager.delete_file(file_path)