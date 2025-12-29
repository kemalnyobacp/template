package main

import "cmp"

// Array and Search Algorithms

func LowerBound[T cmp.Ordered](arr []T, x T) int {
	l, r := 0, len(arr)
	for l < r {
		mid := (l + r) / 2
		if arr[mid] < x {
			l = mid + 1
		} else {
			r = mid
		}
	}
	return l
}

func UpperBound[T cmp.Ordered](arr []T, x T) int {
	l, r := 0, len(arr)
	for l < r {
		mid := (l + r) / 2
		if arr[mid] <= x {
			l = mid + 1
		} else {
			r = mid
		}
	}
	return l
}

func BinarySearch[T Integer](low, high T, check func(T) bool) T {
	for low < high {
		mid := (low + high) / 2
		if check(mid) {
			high = mid
		} else {
			low = mid + 1
		}
	}
	return low
}

func Reverse[T any](s []T) {
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		s[i], s[j] = s[j], s[i]
	}
}

func Sum[T cmp.Ordered](arr []T) T {
	var total T
	for _, v := range arr {
		total += v
	}
	return total
}

func NextPermutation[T cmp.Ordered](arr []T) bool {
	n := len(arr)
	i := n - 2
	for i >= 0 && arr[i] >= arr[i+1] {
		i--
	}
	if i < 0 {
		return false
	}
	j := n - 1
	for arr[j] <= arr[i] {
		j--
	}
	arr[i], arr[j] = arr[j], arr[i]
	Reverse(arr[i+1:])
	return true
}
