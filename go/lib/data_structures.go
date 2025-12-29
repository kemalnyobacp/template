package main

import (
	"cmp"
	"sort"
)

// Disjoint Set Union (Union-Find)
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

// Fenwick Tree (Binary Indexed Tree)
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

// Segment Tree
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

// Min Heap
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

// Max Heap
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

// Stack
type Stack[T any] struct{ data []T }

func (s *Stack[T]) Push(v T) { s.data = append(s.data, v) }
func (s *Stack[T]) Pop() T   { v := s.data[len(s.data)-1]; s.data = s.data[:len(s.data)-1]; return v }
func (s *Stack[T]) Peek() T  { return s.data[len(s.data)-1] }
func (s *Stack[T]) Len() int { return len(s.data) }

// Queue
type Queue[T any] struct{ data []T }

func (q *Queue[T]) Enqueue(v T) { q.data = append(q.data, v) }
func (q *Queue[T]) Dequeue() T  { v := q.data[0]; q.data = q.data[1:]; return v }
func (q *Queue[T]) Len() int    { return len(q.data) }

// Deque
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

// Ordered Set
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
