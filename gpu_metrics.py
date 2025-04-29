import time
import subprocess
import csv
import os

LOG_FILE = "gpu_usage_log.csv"
LOG_INTERVAL = 1  # seconds

def log_gpu_stats():
    write_header = not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "gpu_util_percent", "memory_used_MB"])

        while True:
            try:
                timestamp = int(time.time())
                result = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    encoding="utf-8"
                )
                gpu_util, mem_used = result.strip().split(", ")
                writer.writerow([timestamp, gpu_util, mem_used])
                f.flush()
                time.sleep(LOG_INTERVAL)
            except KeyboardInterrupt:
                print("🛑 Logging stopped by user.")
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")
                time.sleep(LOG_INTERVAL)

if __name__ == "__main__":
    print(f"📈 Logging GPU stats to {LOG_FILE} every {LOG_INTERVAL} second(s)... Press Ctrl+C to stop.")
    log_gpu_stats()
