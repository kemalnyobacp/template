package main

// Mathematical Functions and Number Theory

func AddMod[T Integer](a, b, mod T) T {
	return (a + b) % mod
}

func MulMod[T Integer](a, b, mod T) T {
	return (a * b) % mod
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
