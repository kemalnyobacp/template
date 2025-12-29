package main

import (
	"cmp"
	"slices"
	"sort"
)

var (
	Dir4  = [][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	Dir8  = [][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}
	INF   = int(1e9)
	INF64 = int64(1e18)
	MOD   = int64(1e9 + 7)
	MOD97 = int64(998244353)
)

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

type DSU struct {
	parent []int
	rank   []int
}

func NewDSU(n int) *DSU {
	parent := make([]int, n)
	rank := make([]int, n)
	for i := range parent {
		parent[i] = i
		rank[i] = 1
	}
	return &DSU{parent, rank}
}

func (d *DSU) Find(x int) int {
	if d.parent[x] != x {
		d.parent[x] = d.Find(d.parent[x])
	}
	return d.parent[x]
}

func (d *DSU) Union(x, y int) bool {
	xr, yr := d.Find(x), d.Find(y)
	if xr == yr {
		return false
	}
	if d.rank[xr] < d.rank[yr] {
		xr, yr = yr, xr
	}
	d.parent[yr] = xr
	if d.rank[xr] == d.rank[yr] {
		d.rank[xr]++
	}
	return true
}

type Fenwick[T Integer] struct {
	tree []T
}

func NewFenwick[T Integer](n int) *Fenwick[T] {
	return &Fenwick[T]{make([]T, n+1)}
}

func (f *Fenwick[T]) Update(i int, delta T) {
	for i++; i < len(f.tree); i += i & -i {
		f.tree[i] += delta
	}
}

func (f *Fenwick[T]) Query(i int) T {
	sum := T(0)
	for i++; i > 0; i -= i & -i {
		sum += f.tree[i]
	}
	return sum
}

func (f *Fenwick[T]) RangeQuery(l, r int) T {
	return f.Query(r) - f.Query(l-1)
}

type SegmentTree[T Integer] struct {
	tree []T
	n    int
}

func NewSegmentTree[T Integer](arr []T) *SegmentTree[T] {
	n := len(arr)
	tree := make([]T, 4*n)
	st := &SegmentTree[T]{tree, n}
	st.build(arr, 1, 0, n-1)
	return st
}

func (st *SegmentTree[T]) build(arr []T, node, start, end int) {
	if start == end {
		st.tree[node] = arr[start]
		return
	}
	mid := (start + end) / 2
	st.build(arr, 2*node, start, mid)
	st.build(arr, 2*node+1, mid+1, end)
	st.tree[node] = st.tree[2*node] + st.tree[2*node+1]
}

func (st *SegmentTree[T]) Update(idx int, val T) {
	st.update(1, 0, st.n-1, idx, val)
}

func (st *SegmentTree[T]) update(node, start, end, idx int, val T) {
	if start == end {
		st.tree[node] = val
		return
	}
	mid := (start + end) / 2
	if idx <= mid {
		st.update(2*node, start, mid, idx, val)
	} else {
		st.update(2*node+1, mid+1, end, idx, val)
	}
	st.tree[node] = st.tree[2*node] + st.tree[2*node+1]
}

func (st *SegmentTree[T]) Query(l, r int) T {
	return st.query(1, 0, st.n-1, l, r)
}

func (st *SegmentTree[T]) query(node, start, end, l, r int) T {
	if r < start || end < l {
		return 0
	}
	if l <= start && end <= r {
		return st.tree[node]
	}
	mid := (start + end) / 2
	return st.query(2*node, start, mid, l, r) + st.query(2*node+1, mid+1, end, l, r)
}

type MinHeap[T cmp.Ordered] struct {
	data []T
}

func (h *MinHeap[T]) Len() int           { return len(h.data) }
func (h *MinHeap[T]) Less(i, j int) bool { return h.data[i] < h.data[j] }
func (h *MinHeap[T]) Swap(i, j int)      { h.data[i], h.data[j] = h.data[j], h.data[i] }
func (h *MinHeap[T]) Push(x any)         { h.data = append(h.data, x.(T)) }
func (h *MinHeap[T]) Pop() any {
	old := h.data
	n := len(old)
	x := old[n-1]
	h.data = old[:n-1]
	return x
}

type MaxHeap[T cmp.Ordered] struct {
	data []T
}

func (h *MaxHeap[T]) Len() int           { return len(h.data) }
func (h *MaxHeap[T]) Less(i, j int) bool { return h.data[i] > h.data[j] }
func (h *MaxHeap[T]) Swap(i, j int)      { h.data[i], h.data[j] = h.data[j], h.data[i] }
func (h *MaxHeap[T]) Push(x any)         { h.data = append(h.data, x.(T)) }
func (h *MaxHeap[T]) Pop() any {
	old := h.data
	n := len(old)
	x := old[n-1]
	h.data = old[:n-1]
	return x
}

type Stack[T any] struct{ data []T }

func (s *Stack[T]) Push(v T) { s.data = append(s.data, v) }
func (s *Stack[T]) Pop() T   { v := s.data[len(s.data)-1]; s.data = s.data[:len(s.data)-1]; return v }
func (s *Stack[T]) Peek() T  { return s.data[len(s.data)-1] }
func (s *Stack[T]) Len() int { return len(s.data) }

type Queue[T any] struct{ data []T }

func (q *Queue[T]) Enqueue(v T) { q.data = append(q.data, v) }
func (q *Queue[T]) Dequeue() T  { v := q.data[0]; q.data = q.data[1:]; return v }
func (q *Queue[T]) Len() int    { return len(q.data) }

type Deque[T any] struct {
	data []T
}

func (d *Deque[T]) PushFront(v T) {
	d.data = append([]T{v}, d.data...)
}

func (d *Deque[T]) PushBack(v T) {
	d.data = append(d.data, v)
}

func (d *Deque[T]) PopFront() T {
	v := d.data[0]
	d.data = d.data[1:]
	return v
}

func (d *Deque[T]) PopBack() T {
	v := d.data[len(d.data)-1]
	d.data = d.data[:len(d.data)-1]
	return v
}

func (d *Deque[T]) Front() T {
	return d.data[0]
}

func (d *Deque[T]) Back() T {
	return d.data[len(d.data)-1]
}

func (d *Deque[T]) Len() int {
	return len(d.data)
}

func (d *Deque[T]) Empty() bool {
	return len(d.data) == 0
}

type OrderedSet[T cmp.Ordered] struct {
	data []T
}

func (os *OrderedSet[T]) Insert(x T) {
	i := sort.Search(len(os.data), func(i int) bool { return os.data[i] >= x })
	if i < len(os.data) && os.data[i] == x {
		return
	}
	os.data = append(os.data[:i], append([]T{x}, os.data[i:]...)...)
}

func (os *OrderedSet[T]) Contains(x T) bool {
	i := sort.Search(len(os.data), func(i int) bool { return os.data[i] >= x })
	return i < len(os.data) && os.data[i] == x
}

func (os *OrderedSet[T]) LowerBound(x T) int {
	return sort.Search(len(os.data), func(i int) bool { return os.data[i] >= x })
}

func (os *OrderedSet[T]) UpperBound(x T) int {
	return sort.Search(len(os.data), func(i int) bool { return os.data[i] > x })
}

type Pair[T1, T2 any] struct {
	First  T1
	Second T2
}

func NewMatrix2[T any](n, m int) [][]T {
	mat := make([][]T, n)
	for i := range mat {
		mat[i] = make([]T, m)
	}
	return mat
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

func AddMod[T Integer](a, b, mod T) T {
	return (a + b) % mod
}

func MulMod[T Integer](a, b, mod T) T {
	return (a * b) % mod
}

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

type Integer interface {
	int64 | int32 | int | int8 | int16 | uint | uint8 | uint16 | uint32 | uint64
}
type Signed interface {
	int | int8 | int16 | int32 | int64
}
type Unsigned interface {
	uint | uint8 | uint16 | uint32 | uint64
}

func Gcd[T Integer](a, b T) T {
	if a < 0 {
		a = -a
	}
	if b < 0 {
		b = -b
	}
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func Lcm[T Integer](a, b T) T {
	if a == 0 || b == 0 {
		return 0
	}
	return a / Gcd(a, b) * b
}

func Abs[T Signed](x T) T {
	if x < 0 {
		return -x
	}
	return x
}

func Digits[T Integer](n T) []T {
	var digits []T
	for n > 0 {
		digits = append(digits, n%10)
		n /= 10
	}
	Reverse(digits)
	return digits
}

func Ternary[T any](cond bool, a, b T) T {
	if cond {
		return a
	}
	return b
}

func Manhattan[T Signed](x1, y1, x2, y2 T) T {
	return Abs(x1-x2) + Abs(y1-y2)
}

func NCR[T Integer](n, r T) T {
	if r > n {
		return 0
	}
	if r > n-r {
		r = n - r
	}
	num := T(1)
	den := T(1)
	for i := T(0); i < r; i++ {
		num = (num * (n - i))
		den = (den * (i + 1))
	}
	return num / den
}
func NCRMod[T Integer](n, r, mod T) T {
	if r > n {
		return 0
	}
	if r > n-r {
		r = n - r
	}
	num := T(1)
	den := T(1)
	for i := T(0); i < r; i++ {
		num = (num * (n - i)) % mod
		den = (den * (i + 1)) % mod
	}
	return (num * ModInverse(den, mod)) % mod
}

func Pow[T Integer](base, exp T) T {
	result := T(1)
	for exp > 0 {
		if exp&1 == 1 {
			result = result * base
		}
		base = base * base
		exp >>= 1
	}
	return result
}
func Sieve[T Integer](limit T) []bool {
	isPrime := make([]bool, limit+1)
	for i := T(2); i <= limit; i++ {
		isPrime[i] = true
	}
	for i := T(2); i*i <= limit; i++ {
		if isPrime[i] {
			for j := i * i; j <= limit; j += i {
				isPrime[j] = false
			}
		}
	}
	return isPrime
}

func Factorial[T Integer](n, mod T) T {
	result := T(1)
	for i := T(2); i <= n; i++ {
		result = (result * i) % mod
	}
	return result
}

func IsPrime[T Integer](n T) bool {
	if n <= 1 {
		return false
	}
	if n <= 3 {
		return true
	}
	if n%2 == 0 || n%3 == 0 {
		return false
	}
	for i := T(5); i*i <= n; i += 6 {
		if n%i == 0 || n%(i+2) == 0 {
			return false
		}
	}
	return true
}

func PowMod[T Integer](base, exp, mod T) T {
	result := T(1)
	base %= mod
	for exp > 0 {
		if exp&1 == 1 {
			result = (result * base) % mod
		}
		base = (base * base) % mod
		exp >>= 1
	}
	return result
}

func ModInverse[T Integer](a, mod T) T {
	return PowMod(a, mod-2, mod)
}

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
