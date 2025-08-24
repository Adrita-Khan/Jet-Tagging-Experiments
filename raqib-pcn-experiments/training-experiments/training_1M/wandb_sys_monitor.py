import wandb
import time

def main():
    # Initialize wandb run
    wandb.init(
        project="system-monitoring",
        name="idle-system-monitor",
        config={
            "duration_minutes": 30,
            "purpose": "system metrics only"
        }
    )
    
    # Calculate end time (30 minutes from now)
    duration = 30 * 60  # 30 minutes in seconds
    end_time = time.time() + duration
    
    # Log interval (every 1 second)
    log_interval = 1
    
    try:
        while time.time() < end_time:
            # Log empty dict to keep the run active
            # This minimal logging ensures wandb continues to collect system metrics
            wandb.log({})
            
            # Sleep for the log interval
            time.sleep(log_interval)
    
    except KeyboardInterrupt:
        pass
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()