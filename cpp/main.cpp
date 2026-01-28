#include <bits/stdc++.h>

#define ll long long
#define pb push_back
#define mp make_pair
#define all(x) x.begin(), x.end()
#define f first
#define s second
#define endl '\n'
#define debug(x) cerr << #x << " = " << x << endl;
#define debug2(x, y) cerr << #x << " = " << x << ", " << #y << " = " << y << endl;
#define debug3(x, y, z) cerr << #x << " = " << x << ", " << #y << " = " << y << ", " << #z << " = " << z << endl;
#define file_input "shell.in"
#define file_output "shell.out"
#define usaco 1

using namespace std;

void run() {
    cout << "0" << "\n";
}

void solve() {
    int t;
    cin >> t;
    while (t > 0) {
        run();
        t--;
    }
}

void readInputUsaco() {
    freopen("shell.in", "r", stdin);
}

void writeOutputUsaco() {
    freopen("shell.out", "w", stdout);
}

void runUsaco() {
    readInputUsaco();
    solve();
    writeOutputUsaco();
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    if (usaco) {
        runUsaco();
    } else {
        solve();
    }

    return 0;
}
