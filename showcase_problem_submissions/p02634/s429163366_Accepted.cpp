#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#  include <intrin.h>
#  define __builtin_popcount __popcnt
#endif

#define _USE_MATH_DEFINES

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <stack>
#include <list>
#include <math.h>
#include <bitset>
#include <iterator>
#include <map>
#include <iomanip>
#include <functional>
#include <string>
#include <algorithm>
#include <queue>
#include <random>
#include <numeric>
#include <set>
#include <cassert>
#include <cmath>
#include <cinttypes>

using namespace std;

template<class T> ostream& operator<<(ostream& os, const vector<T>& v) { for (auto i = begin(v); i != end(v); i++) os << *i << (i == end(v) - 1 ? "" : " "); return os; }
template<class T> istream& operator>>(istream& is, vector<T>& v) { for (auto i = begin(v); i != end(v); i++) is >> *i; return is; }
struct fast_io {
	fast_io& operator>> (int& target) { scanf("%" PRId32, &target);  return *this; }
	fast_io& operator>> (unsigned int& target) { scanf("%" PRIu32, &target);  return *this; }
	fast_io& operator>> (long long& target) { scanf("%" PRId64, &target);  return *this; }
	fast_io& operator>> (unsigned long long& target) { scanf("%" PRIu64, &target);  return *this; }
	template<typename T> fast_io& operator>> (vector<T>& target) { for (auto& value : target) { *this >> value; } return *this; }
	fast_io& operator<< (const int value) { printf("%" PRId32, value); return *this; }
	fast_io& operator<< (const unsigned int value) { printf("%" PRIu32, value); return *this; }
	fast_io& operator<< (const long long value) { printf("%" PRId64, value); return *this; }
	fast_io& operator<< (const unsigned long long value) { printf("%" PRIu64, value); return *this; }
	fast_io& operator<< (const string& value) { printf("%s", value.c_str()); return *this; }
	template<typename T> fast_io& operator<< (const vector<T>& target) { for (auto& value : target) { *this << value << " "; } return *this; }
}; fast_io fio;

template<class ForwardIt, class UnaryPredicate>
ForwardIt first_true(ForwardIt first, ForwardIt last, UnaryPredicate predicate) {
	ForwardIt it;
	typename std::iterator_traits<ForwardIt>::difference_type count, step; count = std::distance(first, last);
	while (count > 0) {
		it = first;	step = count / 2;
		std::advance(it, step);
		if (!predicate(*it)) { first = ++it; count -= step + 1; }
		else count = step;
	}
	return first;
}

#define MOD 1000000007
#define pii pair<int,int>
#define pll pair<long long, long long>
#define all(x) (x).begin(), (x).end()
#define sort_all(x) sort(all(x))
#define make_unique(x) (x).erase(unique(all(x)), (x).end())



template<int mod>
class ModNum
{
private:
	int num;

	inline int easy_mod(int value) const
	{
		while (value > mod)
			value -= mod;
		return value;
	}

	inline int int32_normalize(const int value) const
	{
		if (value < 0)
			return mod * (-value / mod + 1);
		else if (value >= mod)
			return value % mod;
		else
			return value;
	}

public:
	ModNum(const int value) { num = int32_normalize(value); }

	ModNum(const long long value) { num = int32_normalize(value % mod); }

	operator int() const { return num; }

	ModNum& operator=(const ModNum<mod>& rhs)
	{
		this->num = rhs.num;
		return *this;
	}

	ModNum operator +(const ModNum& rhs) const { return ModNum(easy_mod(rhs.num + num)); }

	ModNum& operator +=(const ModNum& rhs)
	{
		this->num = easy_mod(num + rhs.num);
		return *this;
	}

	ModNum operator -(const ModNum& rhs) const { return ModNum(easy_mod(num + mod - rhs.num)); }

	ModNum& operator -=(const ModNum& rhs)
	{
		this->num = easy_mod(num + mod - rhs.num);
		return *this;
	}

	ModNum operator *(const int& rhs) const
	{
		long long x = (long long)num * rhs;
		return ModNum(x < mod ? x : (x % mod));
	}

	ModNum& operator *=(const int& rhs)
	{
		long long x = (long long)num * rhs;
		this->num = x < mod ? (int)x : (x % mod);
		return *this;
	}

	ModNum operator /(const ModNum& rhs) const { return div(rhs); }

	ModNum operator /(const int rhs) const { return div(rhs); }

	ModNum div(const ModNum& other) const { return (*this) * (other.pow(mod - 2)); }

	ModNum div(const int& other) const { return div(ModNum(other)); }

	inline ModNum pow(const unsigned long long power) const
	{
		ModNum resp = 1;
		unsigned long long power2_value = 1;
		ModNum power2_mod = *this;
		while (power2_value <= power)
		{
			if (power & power2_value)
				resp *= power2_mod;
			power2_mod *= power2_mod;
			power2_value *= 2ULL;
		}

		return resp;
	}

	inline ModNum pow(const int power) const { return this->pow((const unsigned long long)power); }
};

typedef ModNum<998244353> my_num;


int main() {

#if defined(_DEBUG)
	freopen("input.txt", "r", stdin);
	freopen("output.txt", "w", stdout);
#endif


	int a, b, c, d;
	cin >> a >> b >> c >> d;

	if (b == d)
	{
		my_num dist = d;
		cout << dist.pow(c - a);
		return 0;
	}

	vector<vector<my_num>> dp(d - b + 1 , vector<my_num>(c - a + 1, my_num(0)));
	dp[0][0] = 1;

	for (int i = 1; i <= (d - b); i++)
	{
		dp[i] = dp[i - 1];
		for (int j = 0; j <= (c-a);j++)
		{
			dp[i][j] *= (a + j);
		}
		my_num tmp = 0;
		for (int j = 0; j <= (c - a); j++)
		{
			tmp *= (b + i - 1);
			dp[i][j] += tmp;
			tmp += dp[i - 1][j];
		}
	}

	my_num ans = 0;
	for (int j = 0; j <= (c - a); j++)
	{
		my_num dist = d;
		ans += dp.back()[j] * dist.pow(c-a-j);
	}

	cout << ans;

	return 0;
}