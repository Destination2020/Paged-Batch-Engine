"""Numerical checks for published GPU overlap evidence (no GPU required)."""
import unittest

from summarize_mps_evidence import overlap_summary


def kernel(start, end, pid):
    return dict(start_ns=start, end_ns=end, pid=pid)


class OverlapTest(unittest.TestCase):
    def test_touching_intervals_do_not_overlap(self):
        result = overlap_summary([kernel(0, 5, 1), kernel(5, 10, 2)])
        self.assertEqual(result['cross_process_overlapping_pairs'], 0)
        self.assertEqual(result['cross_process_interval_union_ns'], 0)
        self.assertEqual(result['any_kernel_union_ns'], 10)

    def test_concurrent_streams_are_not_double_counted(self):
        result = overlap_summary([kernel(0, 10, 1), kernel(1, 9, 1), kernel(2, 8, 2)])
        self.assertEqual(result['cross_process_overlapping_pairs'], 2)
        self.assertEqual(result['cross_process_interval_union_ns'], 6)
        self.assertEqual(result['any_kernel_union_ns'], 10)

    def test_idle_gaps_are_excluded(self):
        result = overlap_summary([kernel(0, 5, 1), kernel(2, 4, 2),
                                  kernel(10, 15, 1), kernel(11, 13, 2)])
        self.assertEqual(result['cross_process_interval_union_ns'], 4)
        self.assertEqual(result['any_kernel_union_ns'], 10)

    def test_same_pid_is_not_cross_process_evidence(self):
        result = overlap_summary([kernel(0, 10, 1), kernel(0, 10, 1)])
        self.assertEqual(result['cross_process_overlapping_pairs'], 0)
        self.assertIsNone(result['longest_pair'])


if __name__ == '__main__':
    unittest.main()
