#include <iostream>
#include <bitset>
#include <fstream>
#include <string>
#include <dirent.h>


using namespace std;
/*
这段代码实现了对短整型（short）数据的补码操作，同时保留原数字的符号，返回与原数字绝对值相等但符号位相同的短整型值。
short 在计算机中常常是16位，其中最高位（也就是第15位）被用作符号位。

接下来，我逐行解释这段代码：

short sign = (x >> 15) & 1;：该语句通过无符号右移和与操作获取原数字 x 的符号位。如果 x 为正，返回 0；反之，返回 1。

short original = sign << 15;：original 作为返回结果的初始值，其除符号位外所有位都为 0。

if (sign == 1) {...} else {...}：这是一个条件语句，当符号位为 1（即 x 为负）时执行第一个分支，否则执行第二个分支。

x = ~x + 1;：当 x 为负时，为得到 x 的补码，需要对 x 进行按位取反（~x）后再加 1。

original |= x;：将 original 的符号位和 x 的补码执行按位或操作后赋值给 original。

else {original = x;}：当 x 为正时，original 直接等于 x。

return original;：返回 original。

这段代码能够处理的输入值是在 -32768（也就是最小的 short 值，二进制表示为 1000000000000000）
到32767（最大的 short 值，二进制表示为0111111111111111）之间的任何数。如输入是负数，这段代码将返回该负数的补码并保持符号位不变；
如输入是正数，直接返回该正数。
*/


short complement_to_original_preserve_sign(short x) {
    short sign = (x >> 15) & 1;
    short original = sign << 15;
    if (sign == 1) {
        x = ~x + 1;
        original |= x;
    }
    else {
        original = x;
    }

    return original;
}

int main() {

    // Open  complementary code
    ifstream infile("./Weight_Parameters/Fixed_Point/scnn_layer_3.txt");
    if (!infile) {
        cout << "Failed to open input file." << endl;
        return 1;
    }

    // Create original code
    ofstream outfile("./Weight_Parameters/Fixed_Point/scnn_layer_3_coe.txt");
    if (!outfile) {
        cout << "Failed to create output file." << endl;
        return 1;
    }




    string line;
    while (getline(infile, line)) {
        short x = stoi(line, nullptr, 2);
        short original = complement_to_original_preserve_sign(x);
        outfile << bitset<16>(original) <<"," << endl;
    }

    infile.close();
    outfile.close();

    cout << "Conversion finished." << endl;
    return 0;
}




