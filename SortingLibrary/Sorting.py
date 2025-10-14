import random
import time
from typing import List


class SortingLibrary:
    """A comprehensive library for sorting algorithms with runtime tracking"""

    def __init__(self):
        self.runtime = 0.0
        self._start_time = 0.0

    def _start_timer(self):
        self._start_time = time.perf_counter()

    def _stop_timer(self, arr: List) -> List:
        self.runtime = time.perf_counter() - self._start_time
        return arr

    # ------------------------------------------------------------------
    # --- O(n^2) SIMPLE SORTS ------------------------------------------
    # ------------------------------------------------------------------

    def bubble_sort(self, arr: List) -> List:
        self._start_timer()
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return self._stop_timer(arr)

    def insertion_sort(self, arr: List) -> List:
        self._start_time()
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and key < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return self._stop_timer(arr)

    def selection_sort(self, arr: List) -> List:
        self._start_time()
        n = len(arr)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return self._stop_timer(arr)

    # ------------------------------------------------------------------
    # --- QUICKSORT VARIANTS (O(n log n) Average) ----------------------
    # ------------------------------------------------------------------

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
            self._start_time()

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

    # ------------------------------------------------------------------
    # --- MERGE SORT VARIANTS (O(n log n)) -----------------------------
    # ------------------------------------------------------------------

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
            i += 1
            k += 1
        while j < n2:
            arr[k] = L[j]
            j += 1
            k += 1

    def merge_sort(self, arr: List, l: int = 0, r: int = -1) -> List:
        is_top_level = r == -1
        if is_top_level:
            r = len(arr) - 1
            self._start_time()
            l
        if l < r:
            m = l + (r - l) // 2
            self.merge_sort(arr, l, m, r=-2)
            self.merge_sort(arr, m + 1, r, r=-2)
            self.merge(arr, l, m, r)
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

    # ------------------------------------------------------------------
    # --- NON-COMPARISON SORTS -----------------------------------------
    # ------------------------------------------------------------------

    def counting_sort(self, arr: List) -> List:
        """Implements In_Place Counting_Sort"""
        self._start_time()
        if not arr:
            return self._stop_timer(arr)

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

    def _counting_sort_by_digit(self,arr:List,exp:int,base:int=10):
        """A stable helper for Radix Sort"""
        n=len(arr)
        output=[0]*n
        count=[0]*base
        
        for i in range(n):
            index=(arr[i]//exp)%base
            count[index]+=1
            
        i=n-1
        while i>=0:
            index=(arr[i]//exp)%base
            output[count[index]-1]=arr[i]
            count[index]-=1
            i-=1
        
        for i in range(n):
            arr[i]=output[i]
            
    def radix_sort(self, arr: List, base: int = 10) -> List:
        """Implements Radix Sort (LSD). O(d * (n + b))."""
        self._start_timer()
        if not arr:
            return self._stop_timer(arr)

        max_val = max(arr)
        exp = 1

        while max_val // exp > 0:
            self._counting_sort_by_digit(arr, exp, base)
            exp *= base

    def bucket_sort(self,arr:List,num_buckets=10)->List:
        """Implements Bucket Sort. Average O(n)."""
        self._start_timer()
        if not arr:
            return self._stop_timer(arr)
        
        min_val, max_val = min(arr), max(arr)
        if min_val == max_val:
            return self._stop_timer(arr)
        
        range_val = max_val - min_val
        buckets=[[] for _ in range(num_buckets)]
        bucket_size=range_val/num_buckets
        
        for x in arr:
            idx = min(int(x-min_val)/bucket_size,num_buckets-1)
            buckets[idx].append(x)
            
        sorted_arr_idx=0
        for bucket in buckets:
            self._insertion_sort_sublist(bucket,0,len(bucket)-1)
            
            for elem in bucket:
                arr[sorted_arr_idx]=elem
                sorted_arr_idx+=1
                
        return self._stop_timer(AssertionError)            
      