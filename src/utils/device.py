"""
Device utilities for M2 MacBook Pro - MPS backend support
"""
import torch
import psutil
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def setup_device() -> torch.device:
    """
    Setup and validate device for M2 MacBook Pro training.
    
    Returns:
        torch.device: Configured device (mps, cpu)
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        logger.info("Using Metal Performance Shaders (MPS) backend")
        
        # Set memory fraction to avoid OOM
        try:
            torch.mps.set_per_process_memory_fraction(0.8)
            logger.info("Set MPS memory fraction to 80%")
        except Exception as e:
            logger.warning(f"Could not set MPS memory fraction: {e}")
            
    else:
        device = torch.device("cpu")
        logger.warning("MPS not available, falling back to CPU")
        logger.warning("This will be significantly slower for training")
    
    return device


def get_device_info() -> dict:
    """Get detailed device information for monitoring."""
    info = {
        "device_type": "mps" if torch.backends.mps.is_available() else "cpu",
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built() if torch.backends.mps.is_available() else False,
    }
    
    # System memory info
    memory = psutil.virtual_memory()
    info.update({
        "total_memory_gb": round(memory.total / (1024**3), 2),
        "available_memory_gb": round(memory.available / (1024**3), 2),
        "memory_percent": memory.percent
    })
    
    # CPU info
    info.update({
        "cpu_count": psutil.cpu_count(),
        "cpu_freq_max": psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown"
    })
    
    return info


class ThermalMonitor:
    """Monitor system thermal state for M2 MacBook Pro."""
    
    def __init__(self, temp_threshold: float = 80.0, check_interval: float = 30.0):
        """
        Initialize thermal monitor.
        
        Args:
            temp_threshold: Temperature threshold in Celsius
            check_interval: Check interval in seconds
        """
        self.temp_threshold = temp_threshold
        self.check_interval = check_interval
        self.last_check = 0
        self.thermal_throttling = False
        
    def check_thermal_state(self) -> bool:
        """
        Check if system is thermally throttling.
        
        Returns:
            bool: True if system is cool enough for training
        """
        current_time = time.time()
        if current_time - self.last_check < self.check_interval:
            return not self.thermal_throttling
            
        self.last_check = current_time
        
        try:
            # Get CPU temperature (macOS specific)
            temps = psutil.sensors_temperatures()
            if temps:
                # Get maximum temperature across all sensors
                max_temp = 0
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current > max_temp:
                            max_temp = entry.current
                
                if max_temp > self.temp_threshold:
                    if not self.thermal_throttling:
                        logger.warning(f"High temperature detected: {max_temp:.1f}°C. "
                                     f"Threshold: {self.temp_threshold}°C")
                        self.thermal_throttling = True
                    return False
                else:
                    if self.thermal_throttling:
                        logger.info(f"Temperature normal: {max_temp:.1f}°C. Resuming training.")
                        self.thermal_throttling = False
                    return True
            else:
                # No temperature sensors available, assume OK
                return True
                
        except Exception as e:
            logger.warning(f"Could not read temperature sensors: {e}")
            return True  # Assume OK if we can't read temperature
        
        return True
        
    def wait_for_cooldown(self, max_wait: float = 300.0) -> None:
        """
        Wait for system to cool down.
        
        Args:
            max_wait: Maximum wait time in seconds
        """
        wait_start = time.time()
        
        while not self.check_thermal_state() and (time.time() - wait_start) < max_wait:
            logger.info("Waiting for system to cool down...")
            time.sleep(30)  # Check every 30 seconds
            
        if (time.time() - wait_start) >= max_wait:
            logger.warning(f"System still hot after {max_wait}s, continuing anyway")


def clear_cache():
    """Clear MPS cache to free memory."""
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            logger.debug("Cleared MPS cache")
    except Exception as e:
        logger.warning(f"Could not clear MPS cache: {e}")


class MemoryTracker:
    """Track memory usage during training."""
    
    def __init__(self):
        self.peak_memory = 0
        self.start_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        memory = psutil.virtual_memory()
        return (memory.total - memory.available) / (1024**3)
    
    def update(self) -> dict:
        """Update and return memory statistics."""
        current_memory = self.get_memory_usage()
        self.peak_memory = max(self.peak_memory, current_memory)
        
        return {
            "current_gb": round(current_memory, 2),
            "peak_gb": round(self.peak_memory, 2),
            "delta_gb": round(current_memory - self.start_memory, 2)
        }
    
    def reset_peak(self):
        """Reset peak memory tracking."""
        self.peak_memory = self.get_memory_usage()


# Global instances for convenience
thermal_monitor = ThermalMonitor()
memory_tracker = MemoryTracker()