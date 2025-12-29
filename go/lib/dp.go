package main

import "cmp"

// Dynamic Programming Algorithms

func Knapsack01[T Integer](weights []T, values []T, capacity T) T {
	n := len(weights)
	dp := make([]T, capacity+1)

	for i := range n {
		for w := capacity; w >= weights[i]; w-- {
			dp[w] = Max(dp[w], dp[w-weights[i]]+values[i])
		}
	}
	return dp[capacity]
}

func KnapsackUnbounded[T Integer](weights []T, values []T, capacity T) T {
	n := len(weights)
	dp := make([]T, capacity+1)

	for w := T(1); w <= capacity; w++ {
		for i := range n {
			if weights[i] <= w {
				dp[w] = Max(dp[w], dp[w-weights[i]]+values[i])
			}
		}
	}
	return dp[capacity]
}

func LongestIncreasingSubsequence[T cmp.Ordered](nums []T) int {
	n := len(nums)
	if n == 0 {
		return 0
	}

	dp := make([]int, n)
	maxLen := 1

	for i := range n {
		dp[i] = 1
		for j := range i {
			if nums[i] > nums[j] {
				dp[i] = Max(dp[i], dp[j]+1)
			}
		}
		maxLen = Max(maxLen, dp[i])
	}
	return maxLen
}

func LISBinarySearch[T cmp.Ordered](nums []T) int {
	tails := make([]T, 0)

	for _, num := range nums {
		pos := LowerBound(tails, num)
		if pos == len(tails) {
			tails = append(tails, num)
		} else {
			tails[pos] = num
		}
	}
	return len(tails)
}

func LongestCommonSubsequence(text1, text2 string) int {
	m, n := len(text1), len(text2)
	dp := NewMatrix2[int](m+1, n+1)

	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if text1[i-1] == text2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = Max(dp[i-1][j], dp[i][j-1])
			}
		}
	}
	return dp[m][n]
}

func EditDistance(word1, word2 string) int {
	m, n := len(word1), len(word2)
	dp := NewMatrix2[int](m+1, n+1)

	for i := 0; i <= m; i++ {
		dp[i][0] = i
	}
	for j := 0; j <= n; j++ {
		dp[0][j] = j
	}

	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = 1 + Min3(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
			}
		}
	}
	return dp[m][n]
}

func CoinChange[T Signed](coins []T, amount T) T {
	dp := make([]T, amount+1)
	infT := T(INF)
	Fill(dp, infT)
	dp[0] = 0

	for i := T(1); i <= amount; i++ {
		for _, coin := range coins {
			if coin <= i {
				dp[i] = Min(dp[i], dp[i-coin]+1)
			}
		}
	}
	if dp[amount] == infT {
		return -1
	}
	return dp[amount]
}

func CoinChangeWays[T Integer](coins []T, amount T) T {
	dp := make([]T, amount+1)
	dp[0] = 1

	for _, coin := range coins {
		for i := coin; i <= amount; i++ {
			dp[i] += dp[i-coin]
		}
	}
	return dp[amount]
}

func MaxSubarraySum[T cmp.Ordered](nums []T) T {
	if len(nums) == 0 {
		var zero T
		return zero
	}

	maxEndingHere := nums[0]
	maxSoFar := nums[0]

	for i := 1; i < len(nums); i++ {
		maxEndingHere = Max(nums[i], maxEndingHere+nums[i])
		maxSoFar = Max(maxSoFar, maxEndingHere)
	}
	return maxSoFar
}

func Kadane[T cmp.Ordered](nums []T) (maxSum T, start, end int) {
	if len(nums) == 0 {
		var zero T
		return zero, 0, 0
	}

	maxSum = nums[0]
	currentSum := nums[0]
	startv, endv := 0, 0
	tempStart := 0

	for i := 1; i < len(nums); i++ {
		if nums[i] > currentSum+nums[i] {
			currentSum = nums[i]
			tempStart = i
		} else {
			currentSum += nums[i]
		}

		if currentSum > maxSum {
			maxSum = currentSum
			startv = tempStart
			endv = i
		}
	}

	return maxSum, startv, endv
}

func MaxProductSubarray[T Integer](nums []T) T {
	if len(nums) == 0 {
		var zero T
		return zero
	}

	maxProd := nums[0]
	minProd := nums[0]
	result := nums[0]
	var zero T

	for i := 1; i < len(nums); i++ {
		if nums[i] < zero {
			maxProd, minProd = minProd, maxProd
		}

		maxProd = Max(nums[i], maxProd*nums[i])
		minProd = Min(nums[i], minProd*nums[i])
		result = Max(result, maxProd)
	}
	return result
}

func WordBreak(s string, wordDict []string) bool {
	wordSet := make(map[string]bool)
	for _, word := range wordDict {
		wordSet[word] = true
	}

	n := len(s)
	dp := make([]bool, n+1)
	dp[0] = true

	for i := 1; i <= n; i++ {
		for j := 0; j < i; j++ {
			if dp[j] && wordSet[s[j:i]] {
				dp[i] = true
				break
			}
		}
	}
	return dp[n]
}

func UniquePaths[T Integer](m, n T) T {
	dp := NewMatrix2[T](int(m), int(n))

	for i := T(0); i < m; i++ {
		dp[i][0] = 1
	}
	for j := T(0); j < n; j++ {
		dp[0][j] = 1
	}

	for i := T(1); i < m; i++ {
		for j := T(1); j < n; j++ {
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}
	return dp[m-1][n-1]
}

func UniquePathsWithObstacles[T Integer](grid [][]T) T {
	if len(grid) == 0 || len(grid[0]) == 0 {
		return 0
	}

	m, n := len(grid), len(grid[0])
	dp := NewMatrix2[T](m, n)

	dp[0][0] = Ternary(grid[0][0] == 0, T(1), T(0))

	for i := 1; i < m; i++ {
		if grid[i][0] == 0 {
			dp[i][0] = dp[i-1][0]
		}
	}
	for j := 1; j < n; j++ {
		if grid[0][j] == 0 {
			dp[0][j] = dp[0][j-1]
		}
	}

	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			if grid[i][j] == 0 {
				dp[i][j] = dp[i-1][j] + dp[i][j-1]
			}
		}
	}
	return dp[m-1][n-1]
}

func MinimumPathSum[T Integer](grid [][]T) T {
	if len(grid) == 0 || len(grid[0]) == 0 {
		return 0
	}

	m, n := len(grid), len(grid[0])
	dp := NewMatrix2[T](m, n)
	dp[0][0] = grid[0][0]

	for i := 1; i < m; i++ {
		dp[i][0] = dp[i-1][0] + grid[i][0]
	}
	for j := 1; j < n; j++ {
		dp[0][j] = dp[0][j-1] + grid[0][j]
	}

	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[i][j] = Min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
		}
	}
	return dp[m-1][n-1]
}

func HouseRobber[T cmp.Ordered](nums []T) T {
	n := len(nums)
	if n == 0 {
		var zero T
		return zero
	}
	if n == 1 {
		return nums[0]
	}

	dp := make([]T, n)
	dp[0] = nums[0]
	dp[1] = Max(nums[0], nums[1])

	for i := 2; i < n; i++ {
		dp[i] = Max(dp[i-1], dp[i-2]+nums[i])
	}
	return dp[n-1]
}

func HouseRobberCircle[T cmp.Ordered](nums []T) T {
	n := len(nums)
	if n == 0 {
		var zero T
		return zero
	}
	if n == 1 {
		return nums[0]
	}

	robFirst := HouseRobber(nums[:n-1])
	robLast := HouseRobber(nums[1:])
	return Max(robFirst, robLast)
}

func ClimbStairs[T Integer](n T) T {
	if n <= 2 {
		return n
	}

	dp := make([]T, n+1)
	dp[1] = 1
	dp[2] = 2

	for i := T(3); i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

func DecodeWays(s string) int {
	n := len(s)
	if n == 0 || s[0] == '0' {
		return 0
	}

	dp := make([]int, n+1)
	dp[0] = 1
	dp[1] = 1

	for i := 2; i <= n; i++ {
		oneDigit := int(s[i-1] - '0')
		twoDigits := int(s[i-2]-'0')*10 + oneDigit

		if oneDigit > 0 {
			dp[i] += dp[i-1]
		}
		if twoDigits >= 10 && twoDigits <= 26 {
			dp[i] += dp[i-2]
		}
	}
	return dp[n]
}

func PartitionEqualSubsetSum[T Integer](nums []T) bool {
	total := Sum(nums)
	if total%2 != 0 {
		return false
	}

	target := total / 2
	dp := make([]bool, target+1)
	dp[0] = true

	for _, num := range nums {
		for j := target; j >= num; j-- {
			dp[j] = dp[j] || dp[j-num]
		}
	}
	return dp[target]
}

func NumRollsToTarget[T Integer](n, k, target, mod T) T {
	dp := make([]T, target+1)
	dp[0] = 1

	for i := T(0); i < n; i++ {
		next := make([]T, target+1)
		for t := T(0); t <= target; t++ {
			if dp[t] == 0 {
				continue
			}
			for face := T(1); face <= k && t+face <= target; face++ {
				next[t+face] = (next[t+face] + dp[t]) % mod
			}
		}
		dp = next
	}
	return dp[target]
}

func CountPalindromicSubstrings(s string) int {
	n := len(s)
	dp := NewMatrix2[bool](n, n)
	count := 0

	for i := n - 1; i >= 0; i-- {
		for j := i; j < n; j++ {
			if s[i] == s[j] && (j-i <= 2 || dp[i+1][j-1]) {
				dp[i][j] = true
				count++
			}
		}
	}
	return count
}

func LongestPalindromicSubstring(s string) string {
	n := len(s)
	if n == 0 {
		return ""
	}

	dp := NewMatrix2[bool](n, n)
	start, maxLen := 0, 1

	for i := range n {
		dp[i][i] = true
	}

	for i := 0; i < n-1; i++ {
		if s[i] == s[i+1] {
			dp[i][i+1] = true
			start = i
			maxLen = 2
		}
	}

	for length := 3; length <= n; length++ {
		for i := 0; i <= n-length; i++ {
			j := i + length - 1
			if s[i] == s[j] && dp[i+1][j-1] {
				dp[i][j] = true
				start = i
				maxLen = length
			}
		}
	}

	return s[start : start+maxLen]
}

func MinPathCost[T Integer](grid [][]T, moveCost [][]T) T {
	m, n := len(grid), len(grid[0])
	dp := NewMatrix2[T](m, n)

	for j := range n {
		dp[0][j] = grid[0][j]
	}

	for i := 1; i < m; i++ {
		for j := range n {
			infT := T(INF)
			dp[i][j] = infT
			for k := range n {
				val := dp[i-1][k] + moveCost[grid[i-1][k]][j] + grid[i][j]
				dp[i][j] = Min(dp[i][j], val)
			}
		}
	}

	return MinN(dp[m-1])
}

func MatrixChainMultiplication[T Integer](dims []T) T {
	n := len(dims) - 1
	dp := NewMatrix2[T](n, n)

	for length := 2; length <= n; length++ {
		for i := 0; i <= n-length; i++ {
			j := i + length - 1
			infT := T(INF)
			dp[i][j] = infT
			for k := i; k < j; k++ {
				cost := dp[i][k] + dp[k+1][j] + dims[i]*dims[k+1]*dims[j+1]
				dp[i][j] = Min(dp[i][j], cost)
			}
		}
	}
	return dp[0][n-1]
}

func TSP[T Integer](dist [][]T) T {
	n := len(dist)
	m := 1 << n
	dp := NewMatrix2[T](m, n)
	infT := T(INF)
	for i := range m {
		Fill(dp[i], infT)
	}

	for i := range n {
		dp[1<<i][i] = 0
	}

	for mask := 1; mask < m; mask++ {
		for last := range n {
			if dp[mask][last] == infT {
				continue
			}
			for next := range n {
				if mask>>next&1 == 0 {
					newMask := mask | (1 << next)
					dp[newMask][next] = Min(dp[newMask][next], dp[mask][last]+dist[last][next])
				}
			}
		}
	}

	result := infT
	for i := range n {
		result = Min(result, dp[m-1][i])
	}
	return result
}
