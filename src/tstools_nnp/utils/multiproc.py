"""Module for running functions in parallel on a single node"""

import functools
import math
import os

import psutil
import torch
import torch.multiprocessing as mp


def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def batch_dicts(dicts, num_workers):
    """
    Batch a list of dictionaries into a list of lists of dictionaries, and add a batch number to each dictionary

    Args:
        dicts (list): list of dictionaries
        num_workers (int): number of workers

    Returns:
        batched_dicts (list): list of lists of dictionaries
    """
    batch_size = math.ceil(len(dicts) / num_workers)
    if batch_size == 0:
        batch_size = 1
    batched_dicts = []
    # Batch the dictionaries and add a batch number to each dictionary
    start_index = 0
    for i in range(num_workers):
        if start_index + batch_size > len(dicts):
            for d in dicts[start_index:]:
                d["batch"] = i
            batched_dicts.append(dicts[start_index:])
        else:
            for d in dicts[start_index : start_index + batch_size]:
                d["batch"] = i
            batched_dicts.append(dicts[start_index : start_index + batch_size])
            start_index += batch_size
            # Recalculate the batch size to ensure that the last batch is not empty
            number_of_items_in_batched_dicts = 0
            for dict in batched_dicts:
                number_of_items_in_batched_dicts += len(dict)
            if i != num_workers - 1:
                batch_size = math.ceil((len(dicts) - number_of_items_in_batched_dicts) / (num_workers - (i + 1)))
    # Remove any empty lists
    # batched_dicts = [x for x in batched_dicts if x != []]

    return batched_dicts


def run_func(func, input_list, queue):
    """
    Run a function in parallel with a list of arguments and put the results in a queue.
    Executes in a temporary directory named after the batch number.

    Args:
        func (function): function to run
        input_list (list): list of dictionaries with arguments for the function
        queue (mp.Queue): queue to put the results in

    Returns:
        None
    """
    # Set the number of threads to 1 to avoid issues with torch multiprocessing
    torch.set_num_threads(1)

    # Make and cd into a batch folder to run calculations in
    batch = input_list[0]["batch"]
    os.system("mkdir batch_" + str(batch))
    os.chdir("batch_" + str(batch))

    # Run the function on each input dictionary
    # Remove the batch number from the input dictionary
    for input_dict in input_list:
        del input_dict["batch"]
    results = []
    for input_dict in input_list:
        try:
            result = func(**input_dict)
        except Exception as e:
            print("Error in batch", batch, ":", e)
            continue
        results.append(result)

    # Change directory back to the original directory and remove the batch folder
    os.chdir("..")
    os.system("rm -r batch_" + str(batch))

    # Put the results in the queue
    final_dict = {batch: results}
    queue.put_nowait(final_dict)


def parallel_run_proc(func, input_list, num_workers, calc=None, ase_calc=None):
    """
    Run a function in parallel with a list of arguments

    Returns:
        results (list): list of results from the function
    """
    # Batch the input list
    batched_dicts = batch_dicts(input_list, num_workers)

    # Check which calculators are not None
    if calc is not None and ase_calc is not None:
        func = functools.partial(func, calc=calc, ase_calc=ase_calc)
    elif calc is not None:
        func = functools.partial(func, calc=calc)
    elif ase_calc is not None:
        func = functools.partial(func, ase_calc=ase_calc)

    # Set up the queue and processes
    queue = mp.Queue()
    num_processes = len(batched_dicts)
    processes = []
    rets = []
    for i in range(num_processes):
        p = mp.Process(target=run_func, args=(func, batched_dicts[i], queue))
        p.start()
        processes.append(p)

    for p in processes:
        try:
            ret = queue.get()
        except Exception as e:
            print(f"Error in consumer: {e}")
        rets.append(ret)

    for p in processes:
        p.join()

    queue.close()
    # Sort the results
    new_rets = []
    for i in range(len(rets)):
        for j in range(len(rets)):
            if i == list(rets[j].keys())[0]:
                new_rets.append(rets[j])
                break

    results = []
    for ret in new_rets:
        results.extend(list(ret.values())[0])
    return results
