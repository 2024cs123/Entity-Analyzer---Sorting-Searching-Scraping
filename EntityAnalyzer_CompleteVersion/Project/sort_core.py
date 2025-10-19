import time
import pandas as pd
import random
import numpy as np
from typing import List, Callable, Tuple

class SortingLibrary:
    """A comprehensive library for sorting algorithms with runtime tracking."""

    def __init__(self):
        self.runtime = 0.0
        self._start_time = 0.0

    def _start_timer(self):
        self._start_time = time.perf_counter()

    def _stop_timer(self, arr: List) -> List:
        self.runtime = (time.perf_counter() - self._start_time) * 1000  # Time in ms
        return arr

    # --- Comparison Sorts (O(n^2) and O(n log n)) ---

    def bubble_sort(self, arr: List) -> List:
        self._start_timer()
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j][0] > arr[j + 1][0]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return self._stop_timer(arr)

    def insertion_sort(self, arr: List) -> List:
        self._start_timer()
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and key[0] < arr[j][0]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return self._stop_timer(arr)

    def selection_sort(self, arr: List) -> List:
        self._start_timer()
        n = len(arr)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j][0] < arr[min_idx][0]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return self._stop_timer(arr)

    def _partition_standard(self, arr: List, low: int, high: int) -> int:
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j][0] <= pivot[0]:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    def quicksort_standard(self, arr: List, low: int = 0, high: int = -1) -> List:
        is_top_level = high == -1
        if is_top_level:
            high = len(arr) - 1
            self._start_timer()

        if low < high:
            pivot = self._partition_standard(arr, low, high)
            self.quicksort_standard(arr, low, pivot - 1)
            self.quicksort_standard(arr, pivot + 1, high)
        return self._stop_timer(arr) if is_top_level else arr

    def _partition_randomized(self, arr: List, low: int, high: int) -> int:
        rand_pivot_idx = random.randint(low, high)
        arr[rand_pivot_idx], arr[high] = arr[high], arr[rand_pivot_idx]
        return self._partition_standard(arr, low, high)

    def quicksort_randomized(self, arr: List, low: int = 0, high: int = -1) -> List:
        is_top_level = high == -1
        if is_top_level:
            high = len(arr) - 1
            self._start_timer()

        if low < high:
            pi = self._partition_randomized(arr, low, high)
            self.quicksort_randomized(arr, low, pi - 1)
            self.quicksort_randomized(arr, pi + 1, high)
        return self._stop_timer(arr) if is_top_level else arr

    def _partition_dual_pivot(self, arr: List, low: int, high: int) -> tuple:
        if arr[low][0] > arr[high][0]:
            arr[low], arr[high] = arr[high], arr[low]
        pivot1, pivot2 = arr[low], arr[high]
        leftPivotIndex, rightPivotIndex, iterator = low + 1, high - 1, low + 1

        while iterator <= rightPivotIndex:
            if arr[iterator][0] < pivot1[0]:
                arr[iterator], arr[leftPivotIndex] = arr[leftPivotIndex], arr[iterator]
                leftPivotIndex += 1
            elif arr[iterator][0] >= pivot2[0]:
                while (
                    arr[rightPivotIndex][0] > pivot2[0] and iterator < rightPivotIndex
                ):
                    rightPivotIndex -= 1
                arr[iterator], arr[rightPivotIndex] = (
                    arr[rightPivotIndex],
                    arr[iterator],
                )
                rightPivotIndex -= 1
                if arr[iterator][0] < pivot1[0]:
                    arr[iterator], arr[leftPivotIndex] = (
                        arr[leftPivotIndex],
                        arr[iterator],
                    )
                    leftPivotIndex += 1
            iterator += 1
        rightPivotIndex += 1
        leftPivotIndex -= 1
        arr[low], arr[high], arr[leftPivotIndex], arr[rightPivotIndex] = (
            arr[leftPivotIndex],
            arr[rightPivotIndex],
            arr[low],
            arr[high],
        )
        return leftPivotIndex, rightPivotIndex

    def quicksort_dual_pivot(self, arr: List, low: int = 0, high: int = -1) -> List:
        is_top_level = high == -1
        if is_top_level:
            high = len(arr) - 1
            self._start_timer()

        if low < high:
            lp, gp = self._partition_dual_pivot(arr, low, high)
            self.quicksort_dual_pivot(arr, low, lp - 1)
            self.quicksort_dual_pivot(arr, lp + 1, gp - 1)
            self.quicksort_dual_pivot(arr, gp + 1, high)

        return self._stop_timer(arr) if is_top_level else arr

    def _merge(self, arr: List, l: int, m: int, r: int):
        n1, n2 = m - l + 1, r - m
        L, R = arr[l : l + n1], arr[m + 1 : m + 1 + n2]
        i, j, k = 0, 0, l

        while i < n1 and j < n2:
            if L[i][0] <= R[j][0]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < n1:
            arr[k] = L[i]
            k += 1
            i += 1
        while j < n2:
            arr[k] = R[j]
            k += 1
            j += 1

    def merge_sort(self, arr: List, l: int = 0, r: int = -1) -> List:
        is_top_level = r == -1
        if is_top_level:
            r = len(arr) - 1
            self._start_timer()

        if l < r:
            m = l + (r - l) // 2
            self.merge_sort(arr, l, m)
            self.merge_sort(arr, m + 1, r)
            self._merge(arr, l, m, r)

        return self._stop_timer(arr) if is_top_level else arr

    def _insertion_sort_sublist_tuples(self, arr: List, l: int, r: int):
        """Helper for Insertion Sort on a sub-array of tuples."""
        for i in range(l + 1, r + 1):
            key = arr[i]
            j = i - 1
            while j >= l and key[0] < arr[j][0]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

    def merge_sort_hybrid(
        self, arr: List, l: int = 0, r: int = -1, k: int = 15
    ) -> List:
        is_top_level = r == -1
        if is_top_level:
            r = len(arr) - 1
            self._start_timer()

        if l < r:
            if r - l <= k:
                self._insertion_sort_sublist_tuples(arr, l, r)
            else:
                m = l + (r - l) // 2
                self.merge_sort_hybrid(arr, l, m, k=k)
                self.merge_sort_hybrid(arr, m + 1, r, k=k)
                self._merge(arr, l, m, r)

        return self._stop_timer(arr) if is_top_level else arr


    # --- NON-COMPARISON SORTS (Updated for Negative Integers) ---

    def counting_sort(self, arr: List) -> List:
        self._start_timer()
        if not arr: return self._stop_timer(arr)
        if any(not isinstance(x[0], (int, np.integer)) for x in arr):
            raise TypeError("Counting Sort requires all elements to be integers.")
        
        values = [x[0] for x in arr]
        min_val, max_val = min(values), max(values)
        
        # 1. Calculate shift for negative numbers and range size
        shift = min_val
        range_size = max_val - min_val + 1
        
        count = [0] * range_size
        output = [None] * len(arr)

        # 2. Store count of each element (shifted to be non-negative)
        for value, _ in arr:
            count[value - shift] += 1
            
        # 3. Change count[i] so that count[i] now contains the actual position
        for i in range(1, range_size):
            count[i] += count[i - 1]
            
        # 4. Build the output array using the stable mechanism (reversed iteration)
        for value, index in reversed(arr):
            output_index = count[value - shift] - 1
            output[output_index] = (value, index)
            count[value - shift] -= 1
            
        # 5. Copy the output array to arr
        for i in range(len(arr)):
            arr[i] = output[i]

        return self._stop_timer(arr)

    def _counting_sort_by_digit(self, arr: List, exp: int, base: int = 10):
        n = len(arr)
        output = [None] * n
        count = [0] * base

        for i in range(n):
            index = (arr[i][0] // exp) % base
            count[index] += 1
        for i in range(1, base):
            count[i] += count[i - 1]
        i = n - 1
        while i >= 0:
            index = (arr[i][0] // exp) % base
            output[count[index] - 1] = arr[i]
            count[index] -= 1
            i -= 1
        for i in range(n):
            arr[i] = output[i]

    def radix_sort(self, arr: List, base: int = 10) -> List:
        self._start_timer()
        if not arr: return self._stop_timer(arr)
        if any(not isinstance(x[0], (int, np.integer)) or x[0] < 0 for x in arr):
            raise TypeError(
                "Radix Sort requires all elements to be non-negative integers."
            )

        max_val = max(x[0] for x in arr)
        exp = 1

        while max_val // exp > 0:
            self._counting_sort_by_digit(arr, exp, base)
            exp *= base

        return self._stop_timer(arr)

    def bucket_sort(self, arr: List, num_buckets=10) -> List:
        self._start_timer()
        if not arr: return self._stop_timer(arr)
        values = [x[0] for x in arr]
        if any(not isinstance(x, (int, float, np.number)) for x in values):
            raise TypeError("Bucket Sort requires all elements to be numeric.")

        min_val, max_val = min(values), max(values)
        if min_val == max_val: return self._stop_timer(arr)

        range_val = max_val - min_val
        buckets = [[] for _ in range(num_buckets)]
        bucket_size = range_val / num_buckets if range_val > 0 else 1

        for x_tuple in arr:
            x_val = x_tuple[0]
            idx = int((x_val - min_val) / bucket_size)
            if idx == num_buckets: idx = num_buckets - 1 
            buckets[idx].append(x_tuple)

        sorted_arr_idx = 0
        for bucket in buckets:
            self._insertion_sort_sublist_tuples(bucket, 0, len(bucket) - 1)
            for elem_tuple in bucket:
                arr[sorted_arr_idx] = elem_tuple
                sorted_arr_idx += 1

        return self._stop_timer(arr)


class SortExecutor:
    """Wraps SotingLibrary algorithms to work with Pandas DataFrames."""

    def __init__(self):
        self.library = SortingLibrary()
        self.algorithm_map: dict[str, Tuple[str, Callable]] = {
            # Comparison Sorts (Allow multi-column sort via Pandas fallback)
            "Selection Sort": ("O(n^2)", self.library.selection_sort),
            "Bubble Sort": ("O(n^2)", self.library.bubble_sort),
            "Insertion Sort": ("O(n^2)", self.library.insertion_sort),
            "Quicksort (Standard)": ("O(N log N)", self.library.quicksort_standard),
            "Quicksort (Randomized)": ("O(N log N)", self.library.quicksort_randomized),
            "Quicksort (Dual Pivot)": ("O(N log N)", self.library.quicksort_dual_pivot),
            "Merge Sort": ("O(N log N)", self.library.merge_sort),
            "Merge Sort (Hybrid)": ("O(N log N)", self.library.merge_sort_hybrid),
            
            # Non-Comparison Sorts (Single column only)
            "Counting Sort": ("O(N+K)", self.library.counting_sort),
            "Radix Sort": ("O(d*(N+b))", self.library.radix_sort),
            "Bucket Sort": ("O(N)", self.library.bucket_sort),

            # Optimized
            "Optimized (Pandas Timsort)": ("O(N log N)", self._pandas_sort_stub),
        }
        self.comparison_sorts = [
            "Selection Sort", "Bubble Sort", "Insertion Sort", 
            "Quicksort (Standard)", "Quicksort (Randomized)", 
            "Quicksort (Dual Pivot)", "Merge Sort", "Merge Sort (Hybrid)"
        ]

    def _pandas_sort_stub(self, arr: List) -> List:
        """Stub for the Pandas Timsort execution (used for single-col timing only)."""
        self.library._start_timer()
        self.library.runtime = 0.0
        return arr

    def execute_sort(
        self, df: pd.DataFrame, column_names: List[str], algorithm_name: str
    ) -> Tuple[pd.DataFrame, float, str]:
        if not column_names:
            return df, 0.0, "N/A"

        # 1. Multi-column Sort or explicit Pandas Optimized Sort
        if len(column_names) > 1 or algorithm_name == "Optimized (Pandas Timsort)":
            start_time = time.perf_counter()
            sorted_df = df.sort_values(
                by=column_names, ascending=True, kind="mergesort", ignore_index=True
            )
            end_time = time.perf_counter()
            time_ms = (end_time - start_time) * 1000
            complexity = "O(N log N) Pandas"
            return sorted_df, time_ms, complexity
        
        # 2. Single-column custom algorithm execution

        col_name = column_names[0]
        complexity, sort_func = self.algorithm_map.get(algorithm_name)

        try:
            sort_key_series = (
                pd.to_numeric(df[col_name], errors="coerce")
                if pd.api.types.is_numeric_dtype(df[col_name])
                else df[col_name].astype(str)
            )
        except Exception:
            sort_key_series = df[col_name].astype(str)

        sortable_list = list(zip(sort_key_series.to_list(), df.index.to_list()))

        try:
            sorted_list_of_tuples = sort_func(sortable_list)
            time_ms = self.library.runtime
            sorted_indices = [idx for val, idx in sorted_list_of_tuples]
            sorted_df = df.loc[sorted_indices].reset_index(drop=True)

            return sorted_df, time_ms, complexity

        except TypeError as e:
            raise TypeError(f"Algorithm Mismatch: {e}")

        except Exception as e:
            raise Exception(f"Sort Failed: {e}")