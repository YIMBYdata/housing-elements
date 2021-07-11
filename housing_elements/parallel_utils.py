from tqdm import tqdm
from multiprocessing import Pool

def parallel_process(function, args_list, num_workers=8):
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(function, args_list), total=len(args_list)))
    return results
