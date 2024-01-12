import gc
import unittest

import torch
from bench_group_attention import main as test_entry


def _simple_parse(args_string):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    args_string = args_string.replace('\n', ' ')
    args_string = args_string.strip()
    args = args_string.split(' ')
    args = list(filter(lambda x: x != '', args))
    try:
        test_entry(args, 'test_benchmark_gqa_all')
    except Exception as e:
        print(f'failed with {args_string}')
        raise e
    gc.collect()
    torch.cuda.empty_cache()
    return args


BATCH_TEST = [
    'batch --batch_start 1 --batch_end 256 --batch_num 8 --length 1024 --mean 0.1 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'batch --batch_start 1 --batch_end 256 --batch_num 8 --length 512 --mean 0.1 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'batch --batch_start 1 --batch_end 256 --batch_num 8 --length 2048 --mean 0.1 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'batch --batch_start 1 --batch_end 256 --batch_num 8 --length 1024 --mean 0.2 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'batch --batch_start 1 --batch_end 256 --batch_num 8 --length 512 --mean 0.2 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'batch --batch_start 1 --batch_end 256 --batch_num 8 --length 2048 --mean 0.2 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'batch --batch_start 1 --batch_end 256 --batch_num 8 --length 1024 --mean 0.5 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'batch --batch_start 1 --batch_end 256 --batch_num 8 --length 512 --mean 0.5 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'batch --batch_start 1 --batch_end 256 --batch_num 8 --length 2048 --mean 0.5 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8', ]

LENGTH_TEST = [
    'length --batch 16 --length_start 2 --length_end 3.5 --length_num 10 --length_log10 --mean 0.1 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'length --batch 64 --length_start 2 --length_end 3.5 --length_num 10 --length_log10 --mean 0.1 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'length --batch 128 --length_start 2 --length_end 3.5 --length_num 10 --length_log10 --mean 0.1 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'length --batch 256 --length_start 2 --length_end 3.5 --length_num 10 --length_log10 --mean 0.1 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'length --batch 16 --length_start 2 --length_end 3.5 --length_num 10 --length_log10 --mean 0.2 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'length --batch 64 --length_start 2 --length_end 3.5 --length_num 10 --length_log10 --mean 0.2 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'length --batch 128 --length_start 2 --length_end 3.5 --length_num 10 --length_log10 --mean 0.2 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'length --batch 256 --length_start 2 --length_end 3.5 --length_num 10 --length_log10 --mean 0.2 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'length --batch 16 --length_start 2 --length_end 3.5 --length_num 10 --length_log10 --mean 0.5 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'length --batch 64 --length_start 2 --length_end 3.5 --length_num 10 --length_log10 --mean 0.5 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'length --batch 128 --length_start 2 --length_end 3.5 --length_num 10 --length_log10 --mean 0.5 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'length --batch 256 --length_start 2 --length_end 3.5 --length_num 10 --length_log10 --mean 0.5 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
]
MEAN_TEST = [
    'mean --batch 16 --length 512 --mean_start 0.1 --mean_end 0.95 --mean_num 10 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'mean --batch 64 --length 512 --mean_start 0.1 --mean_end 0.95 --mean_num 10 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'mean --batch 128 --length 512 --mean_start 0.1 --mean_end 0.95 --mean_num 10 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'mean --batch 256 --length 512 --mean_start 0.1 --mean_end 0.95 --mean_num 10 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'mean --batch 16 --length 1024 --mean_start 0.1 --mean_end 0.95 --mean_num 10 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'mean --batch 64 --length 1024 --mean_start 0.1 --mean_end 0.95 --mean_num 10 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'mean --batch 128 --length 1024 --mean_start 0.1 --mean_end 0.95 --mean_num 10 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'mean --batch 256 --length 1024 --mean_start 0.1 --mean_end 0.95 --mean_num 10 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'mean --batch 16 --length 2048 --mean_start 0.1 --mean_end 0.95 --mean_num 10 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'mean --batch 64 --length 2048 --mean_start 0.1 --mean_end 0.95 --mean_num 10 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'mean --batch 128 --length 2048 --mean_start 0.1 --mean_end 0.95 --mean_num 10 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'mean --batch 256 --length 2048 --mean_start 0.1 --mean_end 0.95 --mean_num 10 --std 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
]
STD_TEST = [
    'std --batch 16 --length 512 --std_start 0.1 --std_end 0.95 --std_num 10 --mean 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'std --batch 64 --length 512 --std_start 0.1 --std_end 0.95 --std_num 10 --mean 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'std --batch 128 --length 512 --std_start 0.1 --std_end 0.95 --std_num 10 --mean 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'std --batch 256 --length 512 --std_start 0.1 --std_end 0.95 --std_num 10 --mean 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'std --batch 16 --length 1024 --std_start 0.1 --std_end 0.95 --std_num 10 --mean 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'std --batch 64 --length 1024 --std_start 0.1 --std_end 0.95 --std_num 10 --mean 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'std --batch 128 --length 1024 --std_start 0.1 --std_end 0.95 --std_num 10 --mean 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'std --batch 256 --length 1024 --std_start 0.1 --std_end 0.95 --std_num 10 --mean 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'std --batch 16 --length 2048 --std_start 0.1 --std_end 0.95 --std_num 10 --mean 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'std --batch 64 --length 2048 --std_start 0.1 --std_end 0.95 --std_num 10 --mean 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'std --batch 128 --length 2048 --std_start 0.1 --std_end 0.95 --std_num 10 --mean 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
    'std --batch 256 --length 2048 --std_start 0.1 --std_end 0.95 --std_num 10 --mean 0.05 --outlier 0.02 --chunk 64 --H 64 --h 128 --G 8',
]
OUTLIER_TEST = [
    'outlier --batch 16 --length 512 --mean 0.5 --std 0.05 --outlier_start 0.01 --outlier_end 0.5 --outlier_num 10 --chunk 64 --H 64 --h 128 --G 8',
    'outlier --batch 32 --length 512 --mean 0.5 --std 0.05 --outlier_start 0.01 --outlier_end 0.5 --outlier_num 10 --chunk 64 --H 64 --h 128 --G 8',
    'outlier --batch 64 --length 512 --mean 0.5 --std 0.05 --outlier_start 0.01 --outlier_end 0.5 --outlier_num 10 --chunk 64 --H 64 --h 128 --G 8',
    'outlier --batch 128 --length 512 --mean 0.5 --std 0.05 --outlier_start 0.01 --outlier_end 0.5 --outlier_num 10 --chunk 64 --H 64 --h 128 --G 8',
    'outlier --batch 16 --length 1024 --mean 0.5 --std 0.05 --outlier_start 0.01 --outlier_end 0.5 --outlier_num 10 --chunk 64 --H 64 --h 128 --G 8',
    'outlier --batch 32 --length 1024 --mean 0.5 --std 0.05 --outlier_start 0.01 --outlier_end 0.5 --outlier_num 10 --chunk 64 --H 64 --h 128 --G 8',
    'outlier --batch 64 --length 1024 --mean 0.5 --std 0.05 --outlier_start 0.01 --outlier_end 0.5 --outlier_num 10 --chunk 64 --H 64 --h 128 --G 8',
    'outlier --batch 128 --length 1024 --mean 0.5 --std 0.05 --outlier_start 0.01 --outlier_end 0.5 --outlier_num 10 --chunk 64 --H 64 --h 128 --G 8',
    'outlier --batch 16 --length 2048 --mean 0.5 --std 0.05 --outlier_start 0.01 --outlier_end 0.5 --outlier_num 10 --chunk 64 --H 64 --h 128 --G 8',
    'outlier --batch 32 --length 2048 --mean 0.5 --std 0.05 --outlier_start 0.01 --outlier_end 0.5 --outlier_num 10 --chunk 64 --H 64 --h 128 --G 8',
    'outlier --batch 64 --length 2048 --mean 0.5 --std 0.05 --outlier_start 0.01 --outlier_end 0.5 --outlier_num 10 --chunk 64 --H 64 --h 128 --G 8',
    'outlier --batch 128 --length 2048 --mean 0.5 --std 0.05 --outlier_start 0.01 --outlier_end 0.5 --outlier_num 10 --chunk 64 --H 64 --h 128 --G 8',
]
CHUNK_TEST = [
    'chunk --batch 16 --length 512 --mean 0.5 --std 0.05 --outlier 0.01 --chunk_start 16 --chunk_end 256 --chunk_num 100 --chunk_pot --H 64 --h 128 --G 8',
    'chunk --batch 64 --length 512 --mean 0.5 --std 0.05 --outlier 0.01 --chunk_start 16 --chunk_end 256 --chunk_num 100 --chunk_pot --H 64 --h 128 --G 8',
    'chunk --batch 128 --length 512 --mean 0.5 --std 0.05 --outlier 0.01 --chunk_start 16 --chunk_end 256 --chunk_num 100 --chunk_pot --H 64 --h 128 --G 8',
    'chunk --batch 256 --length 512 --mean 0.5 --std 0.05 --outlier 0.01 --chunk_start 16 --chunk_end 256 --chunk_num 100 --chunk_pot --H 64 --h 128 --G 8',
    'chunk --batch 16 --length 1024 --mean 0.5 --std 0.05 --outlier 0.01 --chunk_start 16 --chunk_end 256 --chunk_num 100 --chunk_pot --H 64 --h 128 --G 8',
    'chunk --batch 64 --length 1024 --mean 0.5 --std 0.05 --outlier 0.01 --chunk_start 16 --chunk_end 256 --chunk_num 100 --chunk_pot --H 64 --h 128 --G 8',
    'chunk --batch 128 --length 1024 --mean 0.5 --std 0.05 --outlier 0.01 --chunk_start 16 --chunk_end 256 --chunk_num 100 --chunk_pot --H 64 --h 128 --G 8',
    'chunk --batch 256 --length 1024 --mean 0.5 --std 0.05 --outlier 0.01 --chunk_start 16 --chunk_end 256 --chunk_num 100 --chunk_pot --H 64 --h 128 --G 8',
    'chunk --batch 16 --length 2048 --mean 0.5 --std 0.05 --outlier 0.01 --chunk_start 16 --chunk_end 256 --chunk_num 100 --chunk_pot --H 64 --h 128 --G 8',
    'chunk --batch 64 --length 2048 --mean 0.5 --std 0.05 --outlier 0.01 --chunk_start 16 --chunk_end 256 --chunk_num 100 --chunk_pot --H 64 --h 128 --G 8',
    'chunk --batch 128 --length 2048 --mean 0.5 --std 0.05 --outlier 0.01 --chunk_start 16 --chunk_end 256 --chunk_num 100 --chunk_pot --H 64 --h 128 --G 8',
    'chunk --batch 256 --length 2048 --mean 0.5 --std 0.05 --outlier 0.01 --chunk_start 16 --chunk_end 256 --chunk_num 100 --chunk_pot --H 64 --h 128 --G 8',
]


class TestBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import warnings
        warnings.simplefilter('ignore', ResourceWarning)

    def test_batch(self):
        for args in BATCH_TEST:
            _simple_parse(args)

    def test_length(self):
        for args in LENGTH_TEST:
            _simple_parse(args)

    def test_mean(self):
        for args in MEAN_TEST:
            _simple_parse(args)

    def test_std(self):
        for args in STD_TEST:
            _simple_parse(args)

    def test_outlier(self):
        for args in OUTLIER_TEST:
            _simple_parse(args)

    def test_chunk(self):
        for args in CHUNK_TEST:
            _simple_parse(args)

    def test_debug(self):
        args = 'chunk --batch 16 --length 512 --mean 0.5 --std 0.05 --outlier 0.01 --chunk_start 16 --chunk_end 256 --chunk_num 100 --chunk_pot --H 64 --h 128 --G 8'
        _simple_parse(args)


def run_test(test_case_class, test_name):
    suite = unittest.TestSuite()
    suite.addTest(test_case_class(test_name))
    runner = unittest.TextTestRunner()
    runner.run(suite)


def find_test_cases(test_case_class):
    return [method for method in dir(test_case_class) if method.startswith('test')]


if __name__ == '__main__':
    import multiprocessing

    test_cases = find_test_cases(TestBenchmark)
    for test in test_cases:
        print(f'running {test}')
        process = multiprocessing.Process(target=run_test, args=(TestBenchmark, test))
        process.start()
        process.join()
