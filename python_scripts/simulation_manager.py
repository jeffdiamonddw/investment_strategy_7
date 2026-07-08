import logging
import multiprocessing as mp
import os
import psutil
import random
import time

import boto3
import numpy as np
from pymoo.core.problem import Problem
from utils import get_dna_hash




# Sentinel to signal workers to shut down
STOP_SIGNAL = "STOP"


class SimulationManager:
    def __init__(
            self, 
            workhorse_cls, 
            workhorse_args,
            num_workers, 
            timeout_sec,
            target_completions_per_batch,
            target_completions_per_generation, 
            recycle_interval = 10,
        
        ):
        
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        

        
        # 2. Setup internal communication
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.workers = []
        self._spawn_workers()

       

        

        self.n_gen = 0

    def _spawn_workers(self):
        """Spins up persistent worker processes."""
        for i in range(self.num_workers):
            p = mp.Process(
                target=self._worker_loop, 
                args=(self.input_queue, self.output_queue, self.workhorse_cls, self.workhorse_args, i),
                daemon=True
            )
            p.start()
            self.workers.append(p)

    @staticmethod
    def _worker_loop(input_queue, output_queue, workhorse_cls, workhorse_args, worker_id):
        """The execution loop running inside each child process."""
        # Initialize the heavy logic locally inside the process
        # This ensures DataFrames are not shared across memory spaces
        #print('debug worker starting', flush = True)
        
       
        
        local_engine = workhorse_cls(**workhorse_args)
       
        while True:
            time.sleep(random.uniform(0, 1))
            task = input_queue.get()
            if task == STOP_SIGNAL:
                break
            
            idx, x_vector = task
            sim_id = get_dna_hash(x_vector)
            #logger.info('{} {} debug got a task'.format(time.time(), sim_id))
            
            # Call the existing _evaluate method from your original class
            result = local_engine.evaluate(x_vector)
            time.sleep(random.uniform(0, 3))
                
            output_queue.put((idx, result, True))
           

    def evaluate(self, X):
        """
        Manager's evaluation logic with an Absolute Generation Deadline.
        Logs generation health and performance metrics to CloudWatch.
        """
        n_individuals = X.shape[0] 
        sim_ids = [get_dna_hash(x) for x in X]
        results_list = [None] * n_individuals
    
       
        
       
        

        
        pending = range(n_individuals)
        total_completions = 0
        while total_completions < self.target_completions_per_generation:
            
            # 2. Push tasks to the fleet
            num_to_submit = min(self.num_workers, len(pending))
            jobs_to_submit = pending[:num_to_submit]
            for idx in jobs_to_submit:
                #print("debug putting on queue", flush = True)
                self.input_queue.put((idx, X[idx]))
                print("put {} on queue at {}".format(sim_ids[idx], time.ctime()))

        
            # 3. Collection Loop
            completions = 0
            batch_start_time = time.time()
            while completions < self.target_completions_per_batch and len(pending) > 0:
                elapsed = time.time() - batch_start_time
                time_remaining = self.timeout_sec - elapsed
                
                if completions >= self.target_completions_per_batch or time_remaining <= 0:
                    break

                
                try:
                    idx, val, success = self.output_queue.get(timeout=max(0.1, time_remaining))
                    sim_id = sim_ids[idx]
                    print("pulled {} off of queue at {}".format(sim_id, time.ctime()))
                    results_list[idx] = val 
                    pending = list(set(pending).difference([idx]))
                    completions += 1
                    total_completions += 1
                except Exception as e:
                    print(e)
                  
            #Empty queue
            time.sleep(1)
            while not self.output_queue.empty():
                try:
                    idx, val, success = self.output_queue.get(timeout=.1)
                    sim_id = sim_ids[idx]
                    print("pulled {} off of queue at {}".format(sim_id, time.ctime()))
                    results_list[idx] = val 
                    pending = list(set(pending).difference([idx]))
                    completions += 1
                    total_completions += 1
                except Exception as e:
                    print(e)
            print('done batch')
            zzz=1
                
            self.force_reset_fleet()

        
        

       
        # 7. The Nuclear Reset
        # We do this AFTER logging so we don't include spawn time in the Gen performance metrics
        self.force_reset_fleet()

        # 8. Finalize output for Pymoo
        self.n_gen += 1
        return results_list
        


    def force_reset_fleet(self):
        """
            Nuclear reset: Terminates all workers, replaces corrupted queues, 
            and respawns the compute fleet to ensure a clean state for the next batch.
        """
        # 1. Kill the existing workers
        # We use terminate() first to allow for a slightly cleaner OS-level cleanup,
        # then follow up with SIGKILL for any stragglers.
        for p in self.workers:
            try:
                if p.is_alive():
                    p.terminate() 
            except Exception:
                pass

        # Give the OS a moment to reap the processes
        time.sleep(0.1)

        # 2. Hard-Kill and Join
        # Ensures no 'zombie' entries remain in the process table
        for p in self.workers:
            try:
                if p.is_alive():
                    os.kill(p.pid, signal.SIGKILL)
                p.join(timeout=0.1)
            except Exception:
                pass

        # 3. Re-create the communication channels
        # Replacing the queues is the only way to guarantee the internal 
        # locks/pipes aren't in a corrupted state from the terminations.
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()

        # 4. Re-spawn the fleet
        self.workers = []
        for i in range(self.num_workers):
            # We pass the NEW queue references here
            p = mp.Process(
                target=self._worker_loop, 
                args=(
                    self.input_queue, 
                    self.output_queue, 
                    self.workhorse_cls, 
                    self.workhorse_args, 
                    i
                ),
                daemon=True
            )
            p.start()
            self.workers.append(p)
        
        # Reset any generation-specific state tracking
        self.prev_target_pos = None

    def cleanup(self):
        """Properly shuts down the worker processes."""
        for _ in range(self.num_workers):
            self.input_queue.put(STOP_SIGNAL)
        for p in self.workers:
            p.join()

    def __getstate__(self):
        """
        Exclude non-pickleable objects (Queues and Processes) 
        from the checkpoint.
        """
        # Create a copy of the object's state to avoid modifying the live object
        state = self.__dict__.copy()
        
        # Replace non-pickleable objects with placeholders
        state['input_queue'] = None
        state['output_queue'] = None
        state['workers'] = []
        
        return state

    def __setstate__(self, state):
        """
        Restore the state from a checkpoint. 
        Note: The actual Queues and Workers are re-initialized in the main() resume logic.
        """
        self.__dict__.update(state)


    def recycle_workers(self):
        print(f"--- Gen {self.n_gen}: Full Fleet Refresh & Pipe Flush ---")
        # 1. Force kill existing processes
        for p in self.workers:
            try:
                os.kill(p.pid, signal.SIGKILL)
                p.join(timeout=0.1)
            except:
                pass
        
        # 2. Flush the Pipes
        for q in [self.input_queue, self.output_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except:
                    break

        # 3. Re-spawn workers
        self.workers = []
        for i in range(self.num_workers):
            p = mp.Process(
                target=self._worker_loop, 
                args=(self.input_queue, self.output_queue, self.workhorse_cls, self.workhorse_args, i),
                daemon=True
            )
            p.start()
            self.workers.append(p)