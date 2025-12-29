package main

import (
	"bufio"
	"os"
	"runtime"
	"runtime/debug"
)

var (
	in      = bufio.NewReader(os.Stdin)
	out     = bufio.NewWriter(os.Stdout)
	println = Println
	scan    = Scan
	print   = Print
)

func Run() {
	Println("0")
}

func Solve() {
	var t int
	Scan(&t)
	for t > 0 {
		Run()
		t--
	}
}

func init() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	debug.SetGCPercent(-1)
}

func main() {
	defer out.Flush()

	Solve()
}
