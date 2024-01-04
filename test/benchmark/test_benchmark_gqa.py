import unittest

from bench_group_attention import main as test_entry


def _simple_parse(args_string):
    args_string = args_string.replace('\n', ' ')
    args_string = args_string.strip()
    args = args_string.split(' ')
    args = list(filter(lambda x: x != '', args))
    return args


class TestBenchmark(unittest.TestCase):
    def test_batch(self):
        args = 'batch --batch_start 1 --batch_end 256 --batch_num 10 --length 1024 --mean 0.2 --std 0.05 --outlier 0.02 --chunk 16 --H 64 --h 128 --G 8'
        args = _simple_parse(args)
        test_entry(args, 'test_benchmark_gqa')
        args = 'batch --batch_start 1 --batch_end 256 --batch_num 10 --length 1024 --mean 0.2 --std 0.05 --outlier 0.02 --chunk 16 --H 32 --h 128 --G 32'
        args = _simple_parse(args)
        test_entry(args, 'test_benchmark_gqa')

    def test_length(self):
        args = 'length --batch 128 --length_start 2 --length_end 3.5 --length_num 10 --length_log10 --mean 0.2 --std 0.05 --outlier 0.02 --chunk 16 --H 64 --h 16 --G 8'
        args = _simple_parse(args)
        test_entry(args, 'test_benchmark_gqa')
        args = 'length --batch 128 --length_start 2 --length_end 3.5 --length_num 10 --length_log10 --mean 0.2 --std 0.05 --outlier 0.02 --chunk 16 --H 32 --h 16 --G 32'
        args = _simple_parse(args)
        test_entry(args, 'test_benchmark_gqa')

    def test_mean(self):
        args = 'mean --batch 128 --length 1024 --mean_start 0.1 --mean_end 0.95 --mean_num 10 --std 0.05 --outlier 0.02 --chunk 16 --H 32 --G 32 --h 16'
        args = _simple_parse(args)
        test_entry(args, 'test_benchmark_gqa')
        args = 'mean --batch 128 --length 1024 --mean_start 0.1 --mean_end 0.95 --mean_num 10 --std 0.05 --outlier 0.02 --chunk 16 --H 64 --G 8 --h 16'
        args = _simple_parse(args)
        test_entry(args, 'test_benchmark_gqa')

    def test_std(self):
        args = 'std --batch 128 --length 1024 --std_start 0.01 --std_end 0.3 --std_num 10 --mean 0.5 --outlier 0.02 --chunk 16 --H 64 --G 8 --h 16'
        args = _simple_parse(args)
        test_entry(args, 'test_benchmark_gqa')
        args = 'std --batch 128 --length 1024 --std_start 0.01 --std_end 0.3 --std_num 10 --mean 0.5 --outlier 0.02 --chunk 16 --H 32 --G 32 --h 16'
        args = _simple_parse(args)
        test_entry(args, 'test_benchmark_gqa')

    def test_outlier(self):
        args = 'outlier --batch 128 --length 1024 --mean 0.5 --std 0.05 --outlier_start 0.01 --outlier_end 0.5 --outlier_num 10 --chunk 16 --H 32 --G 32 --h 16'
        args = _simple_parse(args)
        test_entry(args, 'test_benchmark_gqa')
        args = 'outlier --batch 128 --length 1024 --mean 0.5 --std 0.05 --outlier_start 0.01 --outlier_end 0.5 --outlier_num 10 --chunk 16 --H 64 --G 8 16'
        args = _simple_parse(args)
        test_entry(args, 'test_benchmark_gqa')

    def test_chunk(self):
        args = 'chunk --batch 128 --length 2048 --mean 0.5 --std 0.05 --outlier 0.01 --chunk_start 16 --chunk_end 1024 --chunk_num 100 --chunk_pot --H 64 --G 8 --h 16'
        args = _simple_parse(args)
        test_entry(args, 'test_benchmark_gqa')
        args = 'chunk --batch 128 --length 2048 --mean 0.5 --std 0.05 --outlier 0.01 --chunk_start 16 --chunk_end 1024 --chunk_num 100 --chunk_pot --H 32 --G 32 --h 16'
        args = _simple_parse(args)
        test_entry(args, 'test_benchmark_gqa')


if __name__ == '__main__':
    unittest.main()
