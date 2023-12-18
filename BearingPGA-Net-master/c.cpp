#include <fstream>
#include <iostream>
#include <bitset>
#include <string>

// ... [省略 complement_to_original_preserve_sign 函数代码，保持不变] ...
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
    std::ifstream inFile("scnn_layer_3.txt");
    std::ofstream outFile("scnn_layer_3_new_ww.txt");

    if (!inFile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    short x;
    std::string line;
    while (std::getline(inFile, line)) { // 读取每一行
        x = std::bitset<16>(line).to_ulong(); // 将字符串转换为短整型
        short original = complement_to_original_preserve_sign(x);
        std::bitset<16> bitsetOriginal(original); // 将原始值转换为二进制位集
        outFile << bitsetOriginal.to_string() << "," << std::endl; // 将结果写入文件并附加逗号和换行符
    }

    inFile.close();
    outFile.close();

    std::cout << "Conversion finished." << std::endl;

    return 0;
}
