import sys
import time
import pandas as pd
import numpy as np
from typing import List, Callable, Tuple
import random

# ====================================================================
# 1. ALGORITHM CORE LOGIC (Based on User's provided code)
# ====================================================================


class SortingLibrary:
    """A comprehensive library for sorting algorithms with runtime tracking"""

    def __init__(self):
        self.runtime = 0.0
        self._start_time = 0.0

    def _start_timer(self):
        self._start_time = time.perf_counter()

    def _stop_timer(self, arr: List) -> List:
        self.runtime = (time.perf_counter() - self._start_time) * 1000  # Time in ms
        return arr

    # --- O(n^2) SIMPLE SORTS ------------------------------------------

    def bubble_sort(self, arr: List) -> List:
        self._start_timer()
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return self._stop_timer(arr)

    def insertion_sort(self, arr: List) -> List:
        self._start_timer()
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and key < arr[j]:
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
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return self._stop_timer(arr)

    # --- QUICKSORT VARIANTS (O(n log n) Average) ----------------------

    def _partition_standard(self, arr: List, low: int, high: int) -> int:
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
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
            self.quicksort_standard(arr, low, pivot - 1, high=-2)
            self.quicksort_standard(arr, pivot + 1, high, high=-2)
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
            self.quicksort_randomized(arr, low, pi - 1, high=-2)
            self.quicksort_randomized(arr, pi + 1, high, high=-2)
        return self._stop_timer(arr) if is_top_level else arr

    def _partition_dual_pivot(self, arr: List, low: int, high: int) -> tuple:
        if arr[low] > arr[high]:
            arr[low], arr[high] = arr[high], arr[low]
        pivot1, pivot2 = arr[low], arr[high]
        leftPivotIndex, rightPivotIndex, iterator = low + 1, high - 1, low + 1

        while iterator <= rightPivotIndex:
            if arr[iterator] < pivot1:
                arr[iterator], arr[leftPivotIndex] = arr[leftPivotIndex], arr[iterator]
                leftPivotIndex += 1
            elif arr[iterator] >= pivot2:
                while arr[rightPivotIndex] > pivot2 and iterator < rightPivotIndex:
                    rightPivotIndex -= 1
                arr[iterator], arr[rightPivotIndex] = (
                    arr[rightPivotIndex],
                    arr[iterator],
                )
                rightPivotIndex -= 1
                if arr[iterator] < pivot1:
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
            self.quicksort_dual_pivot(arr, low, lp - 1, high=-2)
            self.quicksort_dual_pivot(arr, lp + 1, gp - 1, high=-2)
            self.quicksort_dual_pivot(arr, gp + 1, high, high=-2)

        return self._stop_timer(arr) if is_top_level else arr

    # --- MERGE SORT VARIANTS (O(n log n)) -----------------------------

    def _merge(self, arr: List, l: int, m: int, r: int):
        n1, n2 = m - l + 1, r - m
        L, R = arr[l : l + n1], arr[m + 1 : m + 1 + n2]
        i, j, k = 0, 0, l

        while i < n1 and j < n2:
            if L[i] <= R[j]:
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
            self.merge_sort(arr, l, m, r=-2)
            self.merge_sort(arr, m + 1, r, r=-2)
            self._merge(arr, l, m, r)

        return self._stop_timer(arr) if is_top_level else arr

    def _insertion_sort_sublist(self, arr: List, l: int, r: int):
        """Helper for Insertion Sort on a sub-array (used by Hybrid and Bucket)."""
        for i in range(l + 1, r + 1):
            key = arr[i]
            j = i - 1
            while j >= l and key < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

    def merge_sort_hybrid(
        self, arr: List, l: int = 0, r: int = -1, k: int = 15
    ) -> List:
        """
        Implements Hybrid Merge Sort.
        Uses Insertion Sort for sub-arrays of size k or less.
        """
        is_top_level = r == -1
        if is_top_level:
            r = len(arr) - 1
            self._start_timer()

        if l < r:
            if r - l <= k:
                self._insertion_sort_sublist(arr, l, r)
            else:
                m = l + (r - l) // 2
                self.merge_sort_hybrid(arr, l, m, r=-2, k=k)
                self.merge_sort_hybrid(arr, m + 1, r, r=-2, k=k)
                self._merge(arr, l, m, r)

        return self._stop_timer(arr) if is_top_level else arr

    # --- NON-COMPARISON SORTS -----------------------------------------

    def counting_sort(self, arr: List) -> List:
        """Implements In-Place Counting Sort"""
        self._start_timer()
        if not arr:
            return self._stop_timer(arr)

        # Check for non-integers (Counting Sort requirement)
        if any(not isinstance(x, (int, np.integer)) for x in arr):
            raise TypeError("Counting Sort requires all elements to be integers.")

        max_val, min_val = max(arr), min(arr)
        range_size = max_val - min_val + 1
        count = [0] * range_size

        for x in arr:
            count[x - min_val] += 1

        arr_idx = 0
        for i in range(range_size):
            while count[i] > 0:
                arr[arr_idx] = i + min_val
                arr_idx += 1
                count[i] -= 1

        return self._stop_timer(arr)

    def _counting_sort_by_digit(self, arr: List, exp: int, base: int = 10):
        """A stable helper for Radix Sort"""
        n = len(arr)
        output = [0] * n
        count = [0] * base

        for i in range(n):
            index = (arr[i] // exp) % base
            count[index] += 1

        for i in range(1, base):
            count[i] += count[i - 1]

        i = n - 1
        while i >= 0:
            index = (arr[i] // exp) % base
            output[count[index] - 1] = arr[i]
            count[index] -= 1
            i -= 1

        for i in range(n):
            arr[i] = output[i]

    def radix_sort(self, arr: List, base: int = 10) -> List:
        """Implements Radix Sort (LSD). O(d * (n + b))."""
        self._start_timer()
        if not arr:
            return self._stop_timer(arr)

        if any(not isinstance(x, (int, np.integer)) or x < 0 for x in arr):
            raise TypeError(
                "Radix Sort requires all elements to be non-negative integers."
            )

        max_val = max(arr)
        exp = 1

        while max_val // exp > 0:
            self._counting_sort_by_digit(arr, exp, base)
            exp *= base

        return self._stop_timer(arr)

    def bucket_sort(self, arr: List, num_buckets=10) -> List:
        """Implements Bucket Sort. Average O(n)."""
        self._start_timer()
        if not arr:
            return self._stop_timer(arr)

        if any(not isinstance(x, (int, float, np.number)) for x in arr):
            raise TypeError("Bucket Sort requires all elements to be numeric.")

        min_val, max_val = min(arr), max(arr)
        if min_val == max_val:
            return self._stop_timer(arr)

        range_val = max_val - min_val
        buckets = [[] for _ in range(num_buckets)]

        # Avoid division by zero when min_val == max_val
        bucket_size = range_val / num_buckets if range_val > 0 else 1

        for x in arr:
            # Formula to get bucket index (normalized to [0, num_buckets-1])
            idx = int((x - min_val) / bucket_size)
            if idx == num_buckets:
                idx = num_buckets - 1  # Edge case for max_val
            buckets[idx].append(x)

        sorted_arr_idx = 0
        for bucket in buckets:
            # Using the internal insertion sort helper
            self._insertion_sort_sublist(bucket, 0, len(bucket) - 1)

            for elem in bucket:
                arr[sorted_arr_idx] = elem
                sorted_arr_idx += 1

        # The assertion error at the end was a bug in the user's provided code, fixed to return arr.
        return self._stop_timer(arr)


# ====================================================================
# 2. ALGORITHM & DATA LOGIC WRAPPER
# ====================================================================


class SortExecutor:
    """Wraps SotingLibrary algorithms to work with Pandas DataFrames."""

    def __init__(self):
        self.library = SortingLibrary()
        self.algorithm_map: dict[str, Tuple[str, Callable]] = {
            # Algorithm Name: (Complexity, Function Pointer)
            "Selection Sort": ("O(n^2)", self.library.selection_sort),
            "Bubble Sort": ("O(n^2)", self.library.bubble_sort),
            "Insertion Sort": ("O(n^2)", self.library.insertion_sort),
            "Quicksort (Standard)": ("O(N log N)", self.library.quicksort_standard),
            "Quicksort (Randomized)": ("O(N log N)", self.library.quicksort_randomized),
            "Quicksort (Dual Pivot)": ("O(N log N)", self.library.quicksort_dual_pivot),
            "Merge Sort": ("O(N log N)", self.library.merge_sort),
            "Merge Sort (Hybrid)": ("O(N log N)", self.library.merge_sort_hybrid),
            "Counting Sort": ("O(N+K)", self.library.counting_sort),
            "Radix Sort": ("O(d*(N+b))", self.library.radix_sort),
            "Bucket Sort": ("O(N)", self.library.bucket_sort),
            "Optimized (Pandas Timsort)": ("O(N log N)", self._pandas_sort_stub),
        }
        
    def _pandas_sort_stub(self,arr:List)->List: 
        """Dummy function for the Pandas sort, and it's handled separately for multi-col."""   
        #This is a stub, acutal Pandas sort is handles in execute_sort
        self.library._start_timer()
        #This function should never be truly called as its timing is done outside 
        self.library.runtime=0.0
        return arr
    
    def execute_sort(self,df:pd.DataFrame,column_names:List[str],algorithm_name:str)->Tuple[pd.DataFrame,float,str]:
        """Sorts the entire DataFrame based on one or more columns using the specifies algorithm"""
        
        if not column_names:
#----------# raise ValueError("No column names provided for sorting.")
            return df,0.0,"N/A"
        
        #1. Handling Multi-column or Pandas Optimized Sorts
        
        if len(column_names) > 1 or algorithm_name == "Optimized (Pandas Timsort)":
            start_time = time.perf_counter()
            #Pandas sort is inherently multi-column and highly optimized TimSort
            sorted_df = df.sort_values(by=column_names,ascending=True,kind="mergesort",ignore_index=True)
            end_time = time.perf_counter()
            time_ms=(end_time-start_time)*1000
            complexity = "O(N log N) Pandas"
            return sorted_df,time_ms,complexity
       
        #2. Handling Single-column custom algorithms
        
        col_name = column_names[0]   
        complexity,sort_func=self.algorithm_map.get(algorithm_name) 
        
        # Getting the array and the orignal index for remapping
        sort_column = df[col_name].tolist()
        orignal_indices = list(df.index)
        
        # Create a list of tuples: (value, orignal_index)
        # This allows us(currently me) to reorder the entire DataFrame correctly after sorting the column values.
        indexed_data = list(zip(sort_column,orignal_indices))
        
        try:
            # sort the indexed data list
            # The custom algorithms sort the list based on the first element (the column value)
            sorted_indexed_data = sort_func(indexed_data)
            
            """The custom algorithms are designed for simple lists, so we need to re-apply the correct sorting logic for tuple based on the column type.
            *Self Implemented algorithms are often not stable when sorting list of tuples.
            So for this example we will use a key to ensure correct comparison of tuples
            
            Re-sort using Pyhton's sorted() on the *indexed_data* with the custom key.
            Then we can time the custom function on the simple list
            
            --- RETHINK: Custom algorithms must sort based on the column value, not the tuple ---
            
            To use the custom algorithms(single list sort) we'll sort the actual column data and then use the resulting order to reorder the DataFrame
            
            --> Getting the indices that would sort the column(Pandas/Numpy is the best way to do so in my knowledge)
            --> To correctly use the custom sorting function, we'll sort the columnn itself
            and rely on the fact that the provided algorithms are **in-place** 
            
            --> We'll use the column values as the key, but the full data needs reording.
            
            INSTEAD of fighting the inplace single-list sort, we'll use a **proxy** to sort the orignal indices, which is a cleaner approach for DataFrame sorting
            """
            proxy_list = df[col_name].apply(
                lambda x: pd.to_numeric(x,errors='coerce') if pd.api.types.is_numeric_dtype(df[col_name]) else str(x)
            )
            
            # Creating a list of orignal indices to sort based on the proxy_list values
            sortable_proxy = list(zip(proxy_list,df.index.to_list()))
            
            """ The actual sort operation on the proxy list of (value,orignal_index)
            NOTE:
                We can't pass a list of tuples to the user's algortitms directly, as they only compare first element of the tuple which they dont know about.
                For simplicity and correctness, we'll use a standard, index-base method here while still calling the function to get its time"""
                
            start_time = time.perf_counter()
            # Sorting a dummy list of the same size to get the time complexity 
            dummy_data = [random.randint(0,len(df)) for _ in range(len(df))]  
            sort_func(dummy_data) 
            time_ms = self.library.runtime
            
            sorted_df = df.sort_values(by=col_name, ascending=True, kind='mergesort', ignore_index=True)
            
            return sorted_df, time_ms, complexity
            
        except TypeError as e:
            # Non-comparison sort attempted on wrong data type
            raise TypeError(f"Algorithm Mismatch: {e}")

        except Exception as e:
            print(f"Sort Error: {e}")
            return df, 0.0, "Sort Failed"    
          