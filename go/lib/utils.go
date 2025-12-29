package main

import (
	"cmp"
	"slices"
)

var (
	Dir4  = [][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	Dir8  = [][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}
	INF   = int(1e9)
	INF64 = int64(1e18)
	MOD   = int64(1e9 + 7)
	MOD97 = int64(998244353)
)

type Integer interface {
	int64 | int32 | int | int8 | int16 | uint | uint8 | uint16 | uint32 | uint64
}
type Signed interface {
	int | int8 | int16 | int32 | int64
}
type Unsigned interface {
	uint | uint8 | uint16 | uint32 | uint64
}

func Fill[T any](s []T, val T) {
	for i := range s {
		s[i] = val
	}
}

func CopySlice[T any](s []T) []T {
	res := make([]T, len(s))
	copy(res, s)
	return res
}

func Any[T any](s []T, f func(T) bool) bool {
	return slices.ContainsFunc(s, f)
}

func All[T any](s []T, f func(T) bool) bool {
	for _, v := range s {
		if !f(v) {
			return false
		}
	}
	return true
}

func Keys[K comparable, V any](m map[K]V) []K {
	keys := make([]K, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

func Values[K comparable, V any](m map[K]V) []V {
	values := make([]V, 0, len(m))
	for _, v := range m {
		values = append(values, v)
	}
	return values
}

func InGrid[T Integer](i, j, n, m T) bool {
	return i >= 0 && i < n && j >= 0 && j < m
}

func SetBit(x, pos int) int {
	return x | (1 << pos)
}

func ClearBit(x, pos int) int {
	return x & ^(1 << pos)
}

func ToggleBit(x, pos int) int {
	return x ^ (1 << pos)
}

func IsSet(x, pos int) bool {
	return (x>>pos)&1 == 1
}

func CountBits(x int) int {
	count := 0
	for x > 0 {
		count += x & 1
		x >>= 1
	}
	return count
}

func LowestSetBit(x int) int {
	return x & -x
}

func CeilDiv[T Integer](a, b T) T {
	return (a + b - 1) / b
}

func SqrtInt[T Integer](x T) T {
	var l, r T = 0, x
	for l <= r {
		mid := (l + r) / 2
		if mid*mid <= x && (mid+1)*(mid+1) > x {
			return mid
		}
		if mid*mid < x {
			l = mid + 1
		} else {
			r = mid - 1
		}
	}
	return l
}

func Range[T Integer](start, end T) []T {
	n := end - start
	res := make([]T, n)
	for i := T(0); i < n; i++ {
		res[i] = start + i
	}
	return res
}

func Asc[T cmp.Ordered](i, j T) bool  { return i < j }
func Desc[T cmp.Ordered](i, j T) bool { return i > j }

func Min[T cmp.Ordered](a, b T) T {
	if a < b {
		return a
	}
	return b
}
func Max[T cmp.Ordered](a, b T) T {
	if a > b {
		return a
	}
	return b
}
func Min3[T cmp.Ordered](a, b, c T) T {
	return Min(a, Min(b, c))
}
func Max3[T cmp.Ordered](a, b, c T) T {
	return Max(a, Max(b, c))
}
func MinN[T cmp.Ordered](vals []T) T {
	if len(vals) == 0 {
		var zero T
		return zero
	}
	min := vals[0]
	for _, v := range vals[1:] {
		if v < min {
			min = v
		}
	}
	return min
}
func MaxN[T cmp.Ordered](vals []T) T {
	if len(vals) == 0 {
		var zero T
		return zero
	}
	max := vals[0]
	for _, v := range vals[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

func Iota[T Integer](n T) []T {
	res := make([]T, n)
	for i := T(0); i < n; i++ {
		res[i] = i
	}
	return res
}

func Ternary[T any](cond bool, a, b T) T {
	if cond {
		return a
	}
	return b
}

func NewMatrix2[T any](n, m int) [][]T {
	mat := make([][]T, n)
	for i := range mat {
		mat[i] = make([]T, m)
	}
	return mat
}

type Pair[T1, T2 any] struct {
	First  T1
	Second T2
}

type Counter[T comparable] map[T]int

func (c Counter[T]) Add(key T, val int) {
	c[key] += val
}

func (c Counter[T]) Inc(key T) {
	c[key]++
}

func (c Counter[T]) Dec(key T) {
	c[key]--
}
